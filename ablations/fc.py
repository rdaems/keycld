import jax
import jax.numpy as jnp
from functools import partial
from types import SimpleNamespace
import wandb
from functools import partial
from simple_parsing import ArgumentParser
import flax.linen as nn

from keycld import train
from keycld.models import generate_gaussian_maps, Block, up, MassMatrixPointMasses, PotentialEnergy, InputMatrix
from keycld.data.dm import Data
from keycld.dm import validate


class EncoderFC(nn.Module):
    num_keypoints: int
    num_hidden_dim: int

    @nn.compact
    def __call__(self, x):
        image_size = x.shape[-2]
        x1 = Block(self.num_hidden_dim)(x)
        down1 = nn.max_pool(x1, (2, 2), (2, 2))

        x2 = Block(2 * self.num_hidden_dim)(down1)
        down2 = nn.max_pool(x2, (2, 2), (2, 2))

        x3 = Block(4 * self.num_hidden_dim)(down2)
        down3 = nn.max_pool(x3, (2, 2), (2, 2))

        x4 = Block(8 * self.num_hidden_dim)(down3)
        x = nn.max_pool(x4, (2, 2), (2, 2))

        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(2 * self.num_keypoints)(x)
        keypoints = x.reshape((x.shape[0], self.num_keypoints, 2))
        h_map = generate_gaussian_maps(keypoints, (image_size, image_size), sigma=0.05)
        return keypoints, h_map


class RendererFC(nn.Module):
    num_hidden_dim: int
    image_size: int = 64

    @nn.compact
    def __call__(self, keypoints):
        batch_size, _, _ = keypoints.shape
        gaussian_maps = generate_gaussian_maps(keypoints, (self.image_size, self.image_size), sigma=0.05)
        x = keypoints.reshape((batch_size, -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(4 * 4 * 8 * self.num_hidden_dim)(x)
        x = x.reshape((batch_size, 4, 4, -1))

        x = Block(8 * self.num_hidden_dim)(x)
        x = up(x)

        x = Block(4 * self.num_hidden_dim)(x)
        x = up(x)

        x = Block(2 * self.num_hidden_dim)(x)
        x = up(x)

        x = Block(self.num_hidden_dim)(x)
        x = up(x)

        x = nn.Conv(3, (3, 3))(x)
        x = nn.sigmoid(x)
        return x, gaussian_maps


class ModelFC:
    def __init__(self, num_keypoints, num_action_dim, num_hidden_dim, static_mass_matrix=True, constraint_fn=None, image_size=64):
        self.num_keypoints = num_keypoints
        self.num_action_dim = num_action_dim
        self.num_hidden_dim = num_hidden_dim
        self.static_mass_matrix = static_mass_matrix
        self.constraint_fn = constraint_fn
        self.image_size = image_size

        self._encoder = EncoderFC(self.num_keypoints, self.num_hidden_dim)
        self._renderer = RendererFC(self.num_hidden_dim, self.image_size)
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


class ExperimentFC(train.ExperimentBase):
    def construct_model(self, data):
        return ModelFC(data.n, data.n, self.num_hidden_dim, static_mass_matrix=True, constraint_fn=data.constraint_fn)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--environment', type=str, help='Which DM control suite environment [pendulum, cartpole, acrobot].')
    parser.add_argument('--init_mode', type=str, help='State initialization mode [rest, random].')
    parser.add_argument('--control', type=str, help='Control mode [yes, no].')
    parser.add_arguments(ExperimentFC, dest='experiment')
    args = parser.parse_args()

    print(args)
    wandb.init(project=f'dm-{args.environment}-fc')
    wandb.config.update(args)
    data = Data(environment=args.environment, init_mode=args.init_mode, control=args.control)

    args.experiment.train(data, validate)
