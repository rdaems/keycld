import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import numpy as onp
import wandb
from functools import partial
from simple_parsing import ArgumentParser
import dataclasses
from tqdm import tqdm

from keycld import train
from keycld.models import KeyCLD, predict_run
from keycld.data.dm import Data
from keycld import util


class ExperimentKeyCLD(train.ExperimentBase):
    def construct_model(self, data):
        num_action_dim = data.n
        return KeyCLD(data.n, num_action_dim, self.num_hidden_dim, True, data.constraint_fn)


def validate(data, model, params, epoch):
    # potential energy
    def calculate_grid_statistics(image):
        keypoint_maps = model.encoder(params, image[None])
        keypoints, keypoint_maps = util.map_to_keypoints(keypoint_maps)
        state = keypoints.flatten()
        if model.constraint_fn:
            constraint_values = model.constraint_fn(state)
        else:
            constraint_values = jnp.zeros(0)
        return model.potential_energy(params, state), constraint_values
    images = data.grid['x']
    positions = data.grid['positions']
    potential_energies, constraint_values = jax.vmap(jax.vmap(calculate_grid_statistics))(images)
    constraint_mean = jnp.mean(constraint_values, axis=(0, 1))
    constraint_std = jnp.std(constraint_values, axis=(0, 1))
    mass_matrix = model.mass_matrix(params, jnp.zeros(2 * data.n))

    # predict some validations set runs
    solver = odeint
    predictions = [predict_run(model, params, run, solver) for run in tqdm(data.val)]
    vpt_mean, vpt_std, vpt_median = util.calculate_vpt(data.epsilon, data.val, predictions)
    print(f'[Epoch {epoch}] VPT: {vpt_mean:.02f} ±{vpt_std:.02f}')

    # generate movies
    visuals = []
    for run, prediction in zip(data.val[:4], predictions[:4]):
        norm = lambda x: (x - x.min(axis=(1,2,3), keepdims=True)) / (x.max(axis=(1,2,3), keepdims=True) - x.min(axis=(1,2,3), keepdims=True))
        x = run['x']
        keypoint_maps = 1. - util.visualize_n_maps(norm(prediction['keypoint_maps']))
        gaussian_maps = 1. - util.visualize_n_maps(norm(prediction['gaussian_maps']))
        x_recon = prediction['x_recon']
        output = (onp.concatenate([x, keypoint_maps, gaussian_maps, x_recon], axis=-2) * 255).clip(0, 255).astype(onp.uint8)
        visuals.append(output)
    visuals = onp.concatenate(visuals, axis=-3)

    wandb.log({
        'constraint_mean': constraint_mean,
        'constraint_std': constraint_std,
        'vpt': vpt_mean,
        'vpt_std': vpt_std,
        'vpt_median': vpt_median,
        'visuals':  wandb.Video(visuals.transpose(0, 3, 1, 2), fps=30, format='gif'),
    }, step=epoch)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--environment', type=str, help='Which DM control suite environment [pendulum, cartpole, acrobot].')
    parser.add_argument('--init_mode', type=str, help='State initialization mode [rest, random].')
    parser.add_argument('--control', type=str, help='Control mode [yes, underactuated, no].')
    parser.add_arguments(ExperimentKeyCLD, dest='experiment')
    args = parser.parse_args()

    print(args)
    wandb.init(project=f'dm-{args.environment}')
    wandb.config.update(args)
    wandb.config.update(dataclasses.asdict(args.experiment))
    data = Data(environment=args.environment, init_mode=args.init_mode, control=args.control)

    args.experiment.train(data, validate)
