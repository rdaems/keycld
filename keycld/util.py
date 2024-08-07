from functools import partial
from itertools import permutations
import torch
import numpy as onp

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_flatten, tree_unflatten


def color_palette(n, s=0.5, offset=0.):
    s = 0.5
    r = min(s / 2, (1 - s) / 2)

    alpha = jnp.linspace(0, 2 * onp.pi, n, endpoint=False) + offset
    alpha = alpha[:, None]
    u = jnp.ones(3) / jnp.sqrt(3)
    v = jnp.array([0., -1., 1.]) / jnp.sqrt(2)
    w = jnp.cross(u, v)

    colors = u + r * jnp.cos(alpha) * v + r * jnp.sin(alpha) * w
    return colors


def visualize_n_maps(x, *args, **kwargs):
    n = x.shape[-1]
    colors = color_palette(n, *args, **kwargs)
    canvas = x[..., jnp.newaxis] * colors
    canvas = canvas.max(axis=-2)
    return canvas


def angle_range(angles):
    return (angles + onp.pi) % (2 * onp.pi) - onp.pi


def get_mesh_grid(shape):
    # defines axis convention
    # (0, 0) at center of image
    # x axis points to the right
    # y axis points upwards
    yy, xx = jnp.meshgrid(
        jnp.linspace(1, -1, shape[-2]),
        jnp.linspace(-1, 1, shape[-1]),
    indexing='ij')
    return xx, yy


def map_to_keypoints(heatmap, softmax=True):
    if softmax:
        heatmap = jnp.exp(heatmap)

    xx, yy = get_mesh_grid(heatmap.shape[-3:-1])

    heatmap_sum = heatmap.sum((-3, -2))
    x = (heatmap * xx[:, :, None]).sum((-3, -2)) / heatmap_sum
    y = (heatmap * yy[:, :, None]).sum((-3, -2)) / heatmap_sum

    keypoints = jnp.stack([x, y], -1)
    return keypoints, heatmap


def generate_gaussian_maps(keypoints, shape, sigma=0.1):
    # keypoints: B x K x 2
    # shape: H x W
    xx, yy = get_mesh_grid(shape)
    m = jnp.exp(- 0.5 * ((xx[None, :, :, None] - keypoints[:, None, None, :, 0]) ** 2 + (yy[None, :, :, None] - keypoints[:, None, None, :, 1]) ** 2) / sigma ** 2)
    return m


def numpy_collate(batch):
    if isinstance(batch[0], onp.ndarray):
        return onp.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    elif isinstance(batch[0], dict):
        return {key: numpy_collate([sample[key] for sample in batch]) for key in batch[0]}
    else:
        return onp.array(batch)


class NumpyLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
              batch_size=batch_size,
              shuffle=shuffle,
              sampler=sampler,
              batch_sampler=batch_sampler,
              num_workers=num_workers,
              collate_fn=numpy_collate,
              pin_memory=pin_memory,
              drop_last=drop_last,
              timeout=timeout,
              worker_init_fn=worker_init_fn)


def augment_angular_dimensions(state, angular_dof_nums):
    augmented_state = []
    for i in range(state.shape[-1]):
        if i in angular_dof_nums:
            augmented_state.append(jnp.cos(state[..., i]))
            augmented_state.append(jnp.sin(state[..., i]))
        else:
            augmented_state.append(state[..., i])
    augmented_state = jnp.stack(augmented_state, -1)
    return augmented_state


def construct_mass_matrix(on_diagonal, off_diagonal):
    n = len(on_diagonal)
    assert (n ** 2 - n) // 2 == len(off_diagonal)

    on_diagonal_idxs = (jnp.arange(n), jnp.arange(n))
    off_diagonal_idxs = jnp.tril_indices(n, -1)

    l = jnp.zeros((n, n))
    l = jax.ops.index_update(l, on_diagonal_idxs, on_diagonal)
    l = jax.ops.index_update(l, off_diagonal_idxs, off_diagonal)

    mass_matrix = l @ l.T
    return mass_matrix


def equation_of_motion(models, params, state, t, action):
    action = params['input_matrix_diagonal'] * action

    def lagrangian(q, q_t):
        M = models['mass_matrix'].apply(params['mass_matrix'], q)
        T = 0.5 * q_t.T @ M @ q_t
        V = models['potential_energy'].apply(params['potential_energy'], q)
        L = T - V
        return L.squeeze()

    q, q_t = jnp.split(state, 2)
    q_tt = jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t)) @ (jax.grad(lagrangian, 0)(q, q_t)
           - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t + action)
    return jnp.concatenate([q_t, q_tt])


def equation_of_motion_constrained(constraint, models, params, state, t, action):
    assert models['mass_matrix'].static
    a = partial(apply, models, params) # shorthand for the models

    x, x_t = jnp.split(state, 2)
    input_matrix = a('input_matrix')(x)

    m_inv = jnp.linalg.pinv(a('mass_matrix')(x))
    f = - jax.jacobian(a('potential_energy'), 0)(x).squeeze() + input_matrix @ action
    Dphi = jax.jacobian(constraint)(x)
    DDphi = jax.jacobian(jax.jacobian(constraint))(x)

    l = jnp.linalg.pinv(Dphi @ m_inv @ Dphi.T) @ (Dphi @ m_inv @ f + DDphi @ x_t @ x_t)
    x_tt = m_inv @ (f - Dphi.T @ l)
    return jnp.concatenate([x_t, x_tt])


def equation_of_motion_constrained_general(constraint, models, params, state, t, action):
    # general formulation, a little bit slower

    def lagrangian(x, x_t):
        M = models['mass_matrix'].apply(params['mass_matrix'], x)
        T = 0.5 * x_t.T @ M @ x_t
        V = models['potential_energy'].apply(params['potential_energy'], x)
        L = T - V
        return L.squeeze()

    x, x_t = jnp.split(state, 2)
    input_matrix = models['input_matrix'].apply(params['input_matrix'], x)

    m_inv = jnp.linalg.pinv(jax.hessian(lagrangian, 1)(x, x_t))
    f = jax.grad(lagrangian, 0)(x, x_t) - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(x, x_t) @ x_t + input_matrix @ action
    Dphi = jax.jacobian(constraint)(x)
    DDphi = jax.jacobian(jax.jacobian(constraint))(x)

    l = jnp.linalg.pinv(Dphi @ m_inv @ Dphi.T) @ (Dphi @ m_inv @ f + DDphi @ x_t @ x_t)
    x_tt = m_inv @ (f - Dphi.T @ l)
    return jnp.concatenate([x_t, x_tt])


def interpolate_bilinear(image, i, j, pad_zero=True, relative_coordinates=True):
    """
    from https://github.com/google/jax/issues/862
    based on http://stackoverflow.com/a/12729229
    
    Interpolate bilinearly at coordinates (i, j) in image.
    image: (height, width)
    rows: (nb_points,) relative coordinates
    cols: (nb_points,) relative coordinates
    returns interpolated values (nb_points,)
    """
    if relative_coordinates:
        i *= image.shape[0]
        j *= image.shape[1]
    if pad_zero:
        image = jnp.pad(image, ((1, 1), (1, 1)))
        i += 1
        j += 1
    height, width = image.shape
    i_0 = jnp.floor(i).astype(jnp.int32)
    i_1 = i_0 + 1
    j_0 = jnp.floor(j).astype(jnp.int32)
    j_1 = j_0 + 1

    def rclip(i): return jnp.clip(i, 0, height - 1)
    def cclip(j): return jnp.clip(j, 0, width - 1)
    Ia = image[rclip(i_0), cclip(j_0)]
    Ib = image[rclip(i_1), cclip(j_0)]
    Ic = image[rclip(i_0), cclip(j_1)]
    Id = image[rclip(i_1), cclip(j_1)]

    wa = (j_1 - j) * (i_1 - i)
    wb = (j_1 - j) * (i - i_0)
    wc = (j - j_0) * (i_1 - i)
    wd = (j - j_0) * (i - i_0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def explicit_euler(f, x0, t, *args):
    dt = t[1] - t[0]
    def f_(x, t):
        x_dot = f(x, t, *args)
        x_new = x + x_dot * dt
        return x_new, x_new
    _, history = lax.scan(f_, x0, t[1:])
    history = jnp.concatenate([x0[None], history])
    return history


def sum_losses(loss_fn, loss_weights):
    def f(*args):
        losses = loss_fn(*args)
        loss = sum([loss_weights[key] * losses[key] for key in losses])
        return loss, losses
    return f


def reduce_mean(fn, axis=0):
    return lambda *args: jax.tree_map(lambda x: jnp.mean(x, axis=axis), fn(*args))


def apply(models, params, key):
    return partial(models[key].apply, params[key])


def loss_fn_batched(loss_fn, params, batch, reduction='mean'):
    assert reduction in [None, 'mean', 'sum']
    value = jax.vmap(loss_fn, in_axes=(None, 0))(params, batch)

    if reduction:
        value_flat, value_tree = tree_flatten(value)
        if reduction == 'mean':
            value_flat_reduced = [jnp.mean(v, axis=0) for v in value_flat]
        elif reduction == 'sum':
            value_flat_reduced = [jnp.sum(v, axis=0) for v in value_flat]
        value = tree_unflatten(value_tree, value_flat_reduced)
    return value


def get_permutations(n):
    return list(permutations(range(n)))


def get_permutated_keypoints(keypoints, axis):
    num_keypoints = keypoints.shape[axis]
    permutated_keypoints = jnp.stack([keypoints.take(p, axis=axis) for p in permutations(range(num_keypoints))])
    return permutated_keypoints


def finite_difference(x, dt, mode='central', order=1):
    # x.shape = (num_timesteps, ...)
    assert mode in ['backward', 'central']
    assert order in [1, 2]
    if order == 2:
        x, x_t = finite_difference(x, dt, mode=mode)
        x_t, x_tt = finite_difference(x_t, dt, mode=mode)
        if mode == 'backward':
            return x[1:], x_t, x_tt
        elif mode == 'central':
            return x[1:-1], x_t, x_tt
    elif mode == 'backward':
        x_t = (x[1:] - x[:-1]) / dt
        return x[1:], x_t
    elif mode == 'central':
        x_t = (x[2:] - x[:-2]) / (2 * dt)
        return x[1:-1], x_t


def calculate_vpt(epsilon, runs, predictions):
    vpts = []
    for run, prediction in zip(runs, predictions):
        mse = onp.mean((run['x'] - prediction['x_recon']) ** 2, axis=(1, 2, 3))
        for vpt, error in enumerate(mse):
            if error > epsilon:
                break
        vpts.append(vpt)
    vpt_mean, vpt_std, vpt_median = onp.mean(vpts), onp.std(vpts), onp.median(vpts)
    return vpt_mean, vpt_std, vpt_median


def project_velocity(constraint_fn, x, x_t):
    if constraint_fn:
        DPhi = jax.jacobian(constraint_fn)(x)
        # return (jnp.eye(len(x)) - DPhi.T @ jnp.linalg.inv(DPhi @ DPhi.T) @ DPhi) @ x_t  # "D. Bertsekas, Nonlinear Programming, 1999"
        return x_t - jnp.linalg.pinv(DPhi) @ DPhi @ x_t  # simpler
    else:
        return x_t


def get_augmentations(random_key):
    random_keys = jax.random.split(random_key, 2)
    rotation = jax.random.choice(random_keys[0], 4).astype(int)
    flip = jax.random.choice(random_keys[1], 2)

    def augment(x):
        x = jnp.rot90(x, rotation, axes=(-3, -2))
        if flip:
            x = x[..., ::-1, :]
        return x

    def unaugment(x):
        if flip:
            x = x[..., ::-1, :]
        x = jnp.rot90(x, - rotation, axes=(-3, -2))
        return x

    return augment, unaugment


def rot90_traceable(m, k=1, axes=(0, 1)):
    k %= 4
    return jax.lax.switch(k, [partial(jnp.rot90, m, k=i, axes=axes) for i in range(4)])


def augment(permutation, x):
    rotation = permutation % 4
    flip = permutation // 4
    x = rot90_traceable(x, rotation, axes=(-3, -2))
    x = jnp.where(flip, x[..., ::-1, :], x)
    return x


def unaugment(permutation, x):
    rotation = permutation % 4
    flip = permutation // 4
    x = jnp.where(flip, x[..., ::-1, :], x)
    x = rot90_traceable(x, - rotation, axes=(-3, -2))
    return x
