import jax
import jax.numpy as jnp
import wandb
from functools import partial
from simple_parsing import ArgumentParser

from keycld import train
from keycld.models import KeyCLD
from keycld.data.dm import Data


class ExperimentKeyCLD(train.ExperimentBase):
    def construct_model(self, data):
        num_action_dim = data.n
        return KeyCLD(data.n, num_action_dim, self.num_hidden_dim, True, data.constraint_fn)


def validate(data, model, epoch):
    # potential energy
    def calculate_potential_energy(image):
        keypoints, _ = model.encoder(image[None])
        state = keypoints.flatten()
        return model.potential_energy(state)
    images = data.grid['x']
    positions = data.grid['positions']
    potential_energies = jax.vmap(jax.vmap(calculate_potential_energy))(images)

    if data.environment == 'pendulum':
        table = wandb.Table(data=[[x, y] for (x, y) in zip(positions[0], potential_energies[0])], columns=['q', 'Potential energy'])
        potential_energy_log = wandb.plot.line(table, 'q', 'Potential energy', title='Potential energy')
    elif data.environment in ['cartpole', 'acrobot']:
        x_labels = range(potential_energies.shape[1])
        y_labels = range(potential_energies.shape[0])
        potential_energy_log = wandb.plots.HeatMap(x_labels, y_labels, potential_energies)

    mass_matrix = model.mass_matrix(jnp.zeros(2 * data.n))
    heatmaps, prediction = train.qualitative_results(data, model, solver='dopri')

    wandb.log({
        'potential_energy': potential_energy_log,
        'mass_matrix': wandb.plots.HeatMap(range(2 * data.n), range(2 * data.n), mass_matrix),
        'heatmaps':  wandb.Video(heatmaps.transpose(0, 3, 1, 2), fps=30, format='gif'),
        'prediction': wandb.Video(prediction.transpose(0, 3, 1, 2), fps=30, format='gif'),
    }, step=epoch)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--environment', type=str, help='Which DM control suite environment [pendulum, cartpole, acrobot].')
    parser.add_argument('--init_mode', type=str, help='State initialization mode [rest, random].')
    parser.add_argument('--control', type=str, help='Control mode [yes, no].')
    parser.add_arguments(ExperimentKeyCLD, dest='experiment')
    args = parser.parse_args()

    print(args)
    wandb.init(project=f'dm-{args.environment}')
    wandb.config.update(args)
    data = Data(environment=args.environment, init_mode=args.init_mode, control=args.control)

    args.experiment.train(data, partial(validate, data))
