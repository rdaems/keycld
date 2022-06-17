import jax
import jax.numpy as jnp
from keycld.models import predict
from keycld.util import apply, finite_difference
from functools import partial


def bce_loss(y_pred, y_true):
    eps = 1e-8
    bce = - jnp.sum(
        y_true * jnp.log(y_pred + eps) + (1 - y_true) * jnp.log(1 - y_pred + eps)
    )
    return bce


def dynamics_loss_finite_diff_fn(ode, t, keypoints, action):
    dt = t[1] - t[0]
    num_timesteps, num_keypoints, _ = keypoints.shape

    x = keypoints.reshape(num_timesteps, -1)
    x, x_t, x_tt = finite_difference(x, dt, mode='central', order=2)

    states = jnp.concatenate(([x, x_t]), axis=1)
    states_t = jax.vmap(lambda state: ode(state, None, action))(states)
    _, x_tt_pred = states_t.split(2, -1)
    loss = jnp.sum((x_tt - x_tt_pred) ** 2) / num_timesteps
    return loss


def loss_fn_separate(dynamics_weight, model, params, item):
    t = item['t']
    x = item['x']
    action = item['action']
    num_timesteps = len(t)
    keypoints, keypoint_maps = model.encoder(params, x)

    dynamics_loss = dynamics_loss_finite_diff_fn(partial(model.ode, params), t, keypoints, action)

    x_recon, gaussian_maps = model.renderer(params, keypoints)
    reconstruction_loss = ((x - x_recon) ** 2).sum() / num_timesteps

    loss = reconstruction_loss + dynamics_weight * dynamics_loss
    aux_output = {
        'reconstruction': reconstruction_loss,
        'dynamics': dynamics_loss,
    }
    return loss, aux_output
