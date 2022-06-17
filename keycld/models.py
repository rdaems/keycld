from functools import partial
from types import SimpleNamespace
import jax
from jax.experimental.ode import odeint
import jax.numpy as jnp
import numpy as onp
import flax.linen as nn
from jax.nn.initializers import variance_scaling, normal
from jax.scipy.optimize import minimize

from keycld.util import construct_mass_matrix, explicit_euler, get_mesh_grid


# kernel_init = variance_scaling(0.01, 'fan_in', 'truncated_normal')
kernel_init = normal(0.01)


class PotentialEnergy(nn.Module):
    num_hidden_dim: int

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 1, 'This module is designed for single use, please use vmap for batching.'
        x = nn.celu(nn.Dense(self.num_hidden_dim, kernel_init=kernel_init)(x))
        x = nn.celu(nn.Dense(self.num_hidden_dim, kernel_init=kernel_init)(x))
        x = nn.Dense(1, kernel_init=kernel_init)(x)
        return x.squeeze()  # return float


class InputMatrix(nn.Module):
    num_action_dim: int
    num_hidden_dim: int

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 1, 'This module is designed for single use, please use vmap for batching.'
        num_dof = len(x)
        x = nn.celu(nn.Dense(self.num_hidden_dim, kernel_init=kernel_init)(x))
        x = nn.celu(nn.Dense(self.num_hidden_dim, kernel_init=kernel_init)(x))
        x = nn.Dense(num_dof * self.num_action_dim, kernel_init=kernel_init)(x)

        x = x.reshape(num_dof, self.num_action_dim)
        return x


class MassMatrix(nn.Module):
    num_hidden_dim: int
    static: bool

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 1, 'This module is designed for single use, please use vmap for batching.'
        num_dof = len(x)
        num_l_elements = (num_dof ** 2 + num_dof) // 2
        if self.static:
            x = self.param('l_elements', normal(), (num_l_elements,))
        else:
            x = nn.celu(nn.Dense(self.num_hidden_dim, kernel_init=kernel_init)(x))
            x = nn.celu(nn.Dense(self.num_hidden_dim, kernel_init=kernel_init)(x))
            x = nn.Dense(num_l_elements, kernel_init=kernel_init)(x)
        on_diagonal = x[..., :num_dof]
        off_diagonal = x[..., num_dof:]
        on_diagonal = nn.softplus(on_diagonal)

        mass_matrix = construct_mass_matrix(on_diagonal, off_diagonal)
        return mass_matrix


class MassMatrixPointMasses(nn.Module):
    @nn.compact
    def __call__(self, x):
        assert x.ndim == 1, 'This module is designed for single use, please use vmap for batching.'
        num_dof = len(x)
        num_keypoints = num_dof // 2
        assert 2 * num_keypoints == num_dof
        point_masses = self.param('point_masses', normal(), (num_keypoints,))
        diagonal = point_masses.repeat(2)
        diagonal = diagonal ** 2
        mass_matrix = jnp.diag(diagonal)
        return mass_matrix


class Block(nn.Module):
    num_hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.num_hidden_dim, (3, 3))(x)
        x = nn.GroupNorm(8)(x)
        x = nn.relu(x)
        return x


def up(x):
    shape = x.shape
    new_shape = [*shape[:-3], 2 * shape[-3], 2 * shape[-2], shape[-1]]
    return jax.image.resize(x, new_shape, 'nearest')


class Encoder(nn.Module):
    num_keypoints: int
    num_hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x1 = Block(self.num_hidden_dim)(x)
        down1 = nn.max_pool(x1, (2, 2), (2, 2))

        x2 = Block(2 * self.num_hidden_dim)(down1)
        down2 = nn.max_pool(x2, (2, 2), (2, 2))

        x3 = Block(4 * self.num_hidden_dim)(down2)
        up3 = up(x3)

        x4 = Block(2 * self.num_hidden_dim)(jnp.concatenate([up3, x2], axis=-1))
        up4 = up(x4)

        x5 = Block(self.num_hidden_dim)(jnp.concatenate([up4, x1], axis=-1))

        x = nn.Conv(self.num_keypoints, (3, 3))(x5)
        h_map = jnp.exp(x) # softmax

        xx, yy = get_mesh_grid(h_map.shape[-3:-1])

        h_map_sum = h_map.sum((-3, -2))
        x = (h_map * xx[:, :, None]).sum((-3, -2)) / h_map_sum
        y = (h_map * yy[:, :, None]).sum((-3, -2)) / h_map_sum

        keypoints = jnp.stack([x, y], -1)
        return keypoints, h_map


def generate_gaussian_maps(keypoints, shape, sigma=0.1):
    # keypoints: B x K x 2
    # shape: H x W
    xx, yy = get_mesh_grid(shape)
    m = jnp.exp(- 0.5 * ((xx[None, :, :, None] - keypoints[:, None, None, :, 0]) ** 2 + (yy[None, :, :, None] - keypoints[:, None, None, :, 1]) ** 2) / sigma ** 2)
    return m


class Renderer(nn.Module):
    num_hidden_dim: int
    image_size: int = 64

    @nn.compact
    def __call__(self, keypoints):
        batch_size, num_keypoints, _ = keypoints.shape
        seed = self.param('seed', normal(), (1, self.image_size, self.image_size, self.num_hidden_dim - num_keypoints))
        gaussian_maps = generate_gaussian_maps(keypoints, (self.image_size, self.image_size))
        x0 = jnp.concatenate([seed.repeat(batch_size, axis=0), gaussian_maps], -1)

        x1 = Block(self.num_hidden_dim)(x0)
        down1 = nn.max_pool(x1, (2, 2), (2, 2))

        x2 = Block(2 * self.num_hidden_dim)(down1)
        down2 = nn.max_pool(x2, (2, 2), (2, 2))

        x3 = Block(4 * self.num_hidden_dim)(down2)
        up3 = up(x3)

        x4 = Block(2 * self.num_hidden_dim)(jnp.concatenate([up3, x2], axis=-1))
        up4 = up(x4)

        x5 = Block(self.num_hidden_dim)(jnp.concatenate([up4, x1], axis=-1))
        x = nn.Conv(3, (3, 3))(x5)
        x = nn.sigmoid(x)
        return x, gaussian_maps


class KeyCLD:
    def __init__(self, num_keypoints, num_action_dim, num_hidden_dim, static_mass_matrix=True, constraint_fn=None, image_size=64):
        self.num_keypoints = num_keypoints
        self.num_action_dim = num_action_dim
        self.num_hidden_dim = num_hidden_dim
        self.static_mass_matrix = static_mass_matrix
        self.constraint_fn = constraint_fn
        self.image_size = image_size

        self._encoder = Encoder(self.num_keypoints, self.num_hidden_dim)
        self._renderer = Renderer(self.num_hidden_dim, self.image_size)
        self._mass_matrix = MassMatrixPointMasses()
        self._potential_energy = PotentialEnergy(self.num_hidden_dim)
        self._input_matrix = InputMatrix(self.num_action_dim, self.num_hidden_dim)

    def init(self, random_key):
        random_keys = jax.random.split(random_key, 5)
        encoder_params = self._encoder.init(random_keys[0], jnp.ones((1, self.image_size, self.image_size, 3)))
        renderer_params = self._renderer.init(random_keys[1], jnp.ones((1, self.num_keypoints, 2)))
        mass_matrix_params = self._mass_matrix.init(random_keys[2], jnp.ones(2 * self.num_keypoints))
        potential_energy_params = self._potential_energy.init(random_keys[3], jnp.ones(2 * self.num_keypoints))
        input_matrix_params = self._input_matrix.init(random_keys[4], jnp.ones(2 * self.num_keypoints))
        params = {
            'encoder': encoder_params,
            'renderer': renderer_params,
            'mass_matrix': mass_matrix_params,
            'potential_energy': potential_energy_params,
            'input_matrix': input_matrix_params,
        }
        return params

    def bind(self, params):
        model = SimpleNamespace()
        model.constraint_fn = self.constraint_fn
        model.encoder = partial(self.encoder, params)
        model.renderer = partial(self.renderer, params)
        model.mass_matrix = partial(self.mass_matrix, params)
        model.potential_energy = partial(self.potential_energy, params)
        model.input_matrix = partial(self.input_matrix, params)
        model.ode = partial(self.ode, params)
        return model

    def encoder(self, params, *args):
        keypoints, keypoint_maps = self._encoder.apply(params['encoder'], *args)
        return keypoints, keypoint_maps

    def renderer(self, params, keypoints):
        return self._renderer.apply(params['renderer'], keypoints)

    def mass_matrix(self, params, *args):
        return self._mass_matrix.apply(params['mass_matrix'], *args)

    def potential_energy(self, params, *args):
        return self._potential_energy.apply(params['potential_energy'], *args)

    def input_matrix(self, params, *args):
        return self._input_matrix.apply(params['input_matrix'], *args)

    def euler_lagrange(self, params, x, x_t, action):
        m_inv = jnp.linalg.pinv(self.mass_matrix(params, x))
        f = - jax.grad(self.potential_energy, 1)(params, x).squeeze() + self.input_matrix(params, x) @ action

        if self.constraint_fn:
            Dphi = jax.jacobian(self.constraint_fn)(x)
            DDphi = jax.jacobian(jax.jacobian(self.constraint_fn))(x)
            l = jnp.linalg.pinv(Dphi @ m_inv @ Dphi.T) @ (Dphi @ m_inv @ f + DDphi @ x_t @ x_t)
            x_tt = m_inv @ (f - Dphi.T @ l)
        else:
            x_tt = m_inv @ f
        return x_tt

    def ode(self, params, state, t, action):
        assert state.ndim == 1, 'This module is designed for single use, please use vmap for batching.'
        x, x_t = jnp.split(state, 2)
        assert len(x) == 2 * self.num_keypoints

        x_tt = self.euler_lagrange(params, x, x_t, action)
        return jnp.concatenate([x_t, x_tt])


def predict(ode, t, keypoints, action, solver='explicit_euler'):
    # keypoints: 2 x K x 2
    assert keypoints.ndim == 3
    assert keypoints.shape[0] == 2
    assert keypoints.shape[2] == 2
    if solver == 'explicit_euler':
        ode_solver = explicit_euler
    elif solver == 'dopri':
        ode_solver = odeint
    num_timesteps = len(t)
    if num_timesteps <= 2:
        return keypoints
    num_keypoints = keypoints.shape[1]
    x_dot = (keypoints[1] - keypoints[0]) / (t[1] - t[0])
    state = jnp.concatenate([keypoints[1].flatten(), x_dot.flatten()])
    history = ode_solver(ode, state, t[1:], action)
    keypoints_pred, _ = history.split(2, -1)
    keypoints_pred = keypoints_pred.reshape((num_timesteps - 1, num_keypoints, 2))
    keypoints_pred = jnp.concatenate([keypoints[0][None], keypoints_pred])
    return keypoints_pred


def predict_constraint(constraint_fn, ode, t, keypoints, action, solver='explicit_euler'):
    if constraint_fn is None:
        return predict(ode, t, keypoints, action, solver)
    # https://stackoverflow.com/questions/23578596/solve-an-implicit-ode-differential-algebraic-equation-dae/23580269#23580269

    # keypoints: 2 x K x 2
    assert keypoints.ndim == 3
    assert keypoints.shape[0] == 2
    assert keypoints.shape[2] == 2
    if solver == 'explicit_euler':
        ode_solver = explicit_euler
    elif solver == 'dopri':
        ode_solver = odeint
    num_timesteps = len(t)
    if num_timesteps <= 2:
        return keypoints
    num_keypoints = keypoints.shape[1]
    x_dot = (keypoints[1] - keypoints[0]) / (t[1] - t[0])
    state = jnp.concatenate([keypoints[1].flatten(), x_dot.flatten()])

    constraint_value = constraint_fn(keypoints[1].flatten())
    constraint_fn_zero = lambda x: ((constraint_fn(x) - constraint_value) ** 2).sum()
    def ode_constraint(state, t, action):
        x, x_t = jnp.split(state, 2)
        result = minimize(constraint_fn_zero, x, method='bfgs')
        state = jnp.concatenate([result.x, x_t])
        return ode(state, t, action)

    history = ode_solver(ode_constraint, state, t[1:], action)
    keypoints_pred, _ = history.split(2, -1)
    keypoints_pred = keypoints_pred.reshape((num_timesteps - 1, num_keypoints, 2))
    keypoints_pred = jnp.concatenate([keypoints[0][None], keypoints_pred])
    return keypoints_pred
