import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from keycld import util
from functools import partial


def bce_loss(y_pred, y_true):
    eps = 1e-4
    bce = - jnp.mean(
        y_true * jnp.log(y_pred + eps) + (1 - y_true) * jnp.log(1 - y_pred + eps)
    )
    return bce


def loss_fn_step(dynamics_weight, num_predicted_steps, model, params, item, augmentation_permutation, bce_weight=1.):
    solver = partial(odeint, mxstep=2000)

    t = item['t']
    dt = t[1] - t[0]
    z = item['x']
    action = item['action']
    num_timesteps = len(t)
    t_pred = jnp.arange(num_predicted_steps + 1) * dt

    keypoint_maps = model.encoder(params, jax.vmap(util.augment)(augmentation_permutation, z))
    keypoints, keypoint_maps = util.map_to_keypoints(jax.vmap(util.unaugment)(augmentation_permutation, keypoint_maps))
    x = keypoints.reshape(num_timesteps, -1)
    x, x_t = util.finite_difference(x, dt)
    x_t = jax.vmap(partial(util.project_velocity, model.constraint_fn))(x, x_t)

    state_inits = jnp.concatenate([x, x_t], axis=-1)[:- (num_predicted_steps + 1)]  # crop at the end if predicted steps > 1, for these predictions we have no ground truth
    state_preds = jax.vmap(solver, in_axes=(None, 0, None, None))(partial(model.ode, params), state_inits, t_pred, action)
    x_preds, _ = jnp.split(state_preds, 2, -1)
    dynamics_loss = 0.
    for i, x_pred in enumerate(x_preds):
        dynamics_loss += ((x[i:i+num_predicted_steps+1] - x_pred) ** 2).sum()

    z_recon, gaussian_maps = model.renderer(params, keypoints)
    reconstruction_loss = ((z - z_recon) ** 2).sum() / num_timesteps

    keypoint_maps_n = keypoint_maps / keypoint_maps.max(axis=(1, 2), keepdims=True)
    bce_loss_value = bce_loss(keypoint_maps_n, jax.lax.stop_gradient(gaussian_maps))

    loss = reconstruction_loss + dynamics_weight * dynamics_loss + bce_weight * bce_loss_value
    aux_output = {
        'reconstruction': reconstruction_loss,
        'dynamics': dynamics_loss,
        'bce_loss': bce_loss_value,
    }
    return loss, aux_output
