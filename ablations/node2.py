import jax
import jax.numpy as jnp
from functools import partial
from types import SimpleNamespace
import wandb
from simple_parsing import ArgumentParser
import flax.linen as nn

from keycld import train
from keycld.models import Encoder, Renderer, kernel_init
from keycld.data.dm import Data


class ODE(nn.Module):
    num_hidden_dim: int

    @nn.compact
    def __call__(self, x, x_t, u):
        assert x.ndim == 1, 'This module is designed for single use, please use vmap for batching.'
        num_state_dim = len(x)

        inputs = jnp.concatenate([x, x_t, u])
        h = nn.celu(nn.Dense(2 * self.num_hidden_dim, kernel_init=kernel_init)(inputs))
        h = nn.celu(nn.Dense(2 * self.num_hidden_dim, kernel_init=kernel_init)(h))
        h = nn.celu(nn.Dense(2 * self.num_hidden_dim, kernel_init=kernel_init)(h))
        x_tt = nn.Dense(num_state_dim, kernel_init=kernel_init)(h)
        return x_tt


class ModelNODE2:
    def __init__(self, num_keypoints, num_hidden_dim, image_size=64):
        self.num_keypoints = num_keypoints
        self.num_hidden_dim = num_hidden_dim
        self.image_size = image_size
        self.constraint_fn = None

        self._ode = ODE(self.num_hidden_dim)
        self._encoder = Encoder(self.num_keypoints, self.num_hidden_dim)
        self._renderer = Renderer(self.num_hidden_dim, self.image_size)

    def init(self, random_key):
        random_keys = jax.random.split(random_key, 3)
        encoder_params = self._encoder.init(random_keys[0], jnp.ones((1, self.image_size, self.image_size, 3)))
        renderer_params = self._renderer.init(random_keys[1], jnp.ones((1, self.num_keypoints, 2)))
        ode_params = self._ode.init(random_keys[2], jnp.ones(2 * self.num_keypoints), jnp.ones(2 * self.num_keypoints), jnp.ones(self.num_keypoints))
        params = {
            'encoder': encoder_params,
            'renderer': renderer_params,
            'ode': ode_params,
        }
        return params

    def bind(self, params):
        model = SimpleNamespace()
        model.constraint_fn = self.constraint_fn
        model.encoder = partial(self.encoder, params)
        model.renderer = partial(self.renderer, params)
        model.ode = partial(self.ode, params)
        return model

    def encoder(self, params, *args):
        keypoints, keypoint_maps = self._encoder.apply(params['encoder'], *args)
        return keypoints, keypoint_maps

    def renderer(self, params, keypoints):
        return self._renderer.apply(params['renderer'], keypoints)

    def ode(self, params, state, t, action):
        assert state.ndim == 1, 'This module is designed for single use, please use vmap for batching.'
        x, x_t = jnp.split(state, 2)
        assert len(x) == 2 * self.num_keypoints

        x_tt = self._ode.apply(params['ode'], x, x_t, action)
        return jnp.concatenate([x_t, x_tt])


class ExperimentNODE2(train.ExperimentBase):
    def construct_model(self, data):
        return ModelNODE2(data.n, self.num_hidden_dim)


def validate(model, epoch):
    heatmaps, prediction = train.qualitative_results(data, model, solver='dopri')

    wandb.log({
        'heatmaps':  wandb.Video(heatmaps.transpose(0, 3, 1, 2), fps=30, format='gif'),
        'prediction': wandb.Video(prediction.transpose(0, 3, 1, 2), fps=30, format='gif'),
    }, step=epoch)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--environment', type=str, help='Which DM control suite environment [pendulum, cartpole, acrobot].')
    parser.add_argument('--init_mode', type=str, help='State initialization mode [rest, random].')
    parser.add_argument('--control', type=str, help='Control mode [yes, no].')
    parser.add_arguments(ExperimentNODE2, dest='experiment')
    args = parser.parse_args()

    print(args)
    wandb.init(project=f'dm-{args.environment}-node2')
    wandb.config.update(args)
    data = Data(environment=args.environment, init_mode=args.init_mode, control=args.control)

    args.experiment.train(data, validate)
