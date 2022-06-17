import os
import pickle
import jax
from tqdm import tqdm
import optax
import numpy as onp
import matplotlib.pyplot as plt
from functools import partial
import wandb
from dataclasses import dataclass

from keycld.losses import loss_fn_separate
from keycld.util import reduce_mean, NumpyLoader
from keycld.models import predict_constraint


@dataclass
class ExperimentBase:
    num_epochs: int         # Number of training epochs.
    learning_rate: float    # Learning rate.
    batch_size: int         # Batch size.
    num_hidden_dim: int     # Number of hidden layers in models.
    dynamics_weight: float  # Weight factor of dynamics loss.

    def configure_optimizers(self, params):
        self.tx = optax.adam(self.learning_rate)
        self.opt_state = self.tx.init(params)

    def update(self, params, grads):
        updates, opt_state = self.tx.update(grads, self.opt_state)
        params = optax.apply_updates(params, updates)
        return params

    def construct_model(self, data):
        raise NotImplementedError

    def train(self, data, validate_fn):
        dataloader = NumpyLoader(data.train, batch_size=self.batch_size, num_workers=12, shuffle=True)

        model = self.construct_model(data)
        params = model.init(jax.random.PRNGKey(1))
        self.configure_optimizers(params)

        loss_fn = partial(loss_fn_separate, self.dynamics_weight, model)
        loss_grad_fn = jax.jit(jax.value_and_grad(reduce_mean(jax.vmap(loss_fn, in_axes=(None, 0))), has_aux=True))
        for epoch in range(self.num_epochs):
            total_loss, total_loss_aux = [], []
            with tqdm(total=len(dataloader)) as pbar:
                for batch in dataloader:
                    (loss_val, loss_aux), grads = loss_grad_fn(params, batch)
                    params = self.update(params, grads)
                    total_loss.append(loss_val)
                    total_loss_aux.append(loss_aux)

                    pbar.set_description(f'[Epoch {epoch}] Loss: {loss_val:.04f}, ' + ', '.join([f'{key}: {value:.04f}' for key, value in loss_aux.items()]))
                    pbar.update(1)
                total_loss = onp.mean(total_loss)
                total_loss_aux = {key: onp.mean([d[key] for d in total_loss_aux]) for key in total_loss_aux[0]}
                wandb.log({'loss': total_loss}, step=epoch)
                wandb.log(total_loss_aux, step=epoch)
                pbar.set_description(f'[Epoch {epoch}] Loss: {loss_val:.04f}, ' + ', '.join([f'{key}: {value:.04f}' for key, value in total_loss_aux.items()]))

            validate_fn(model.bind(params), epoch)
        self.save_model(model.bind(params))

    def save_model(self, model, path=None):
        if path is None:
            path = os.path.join(wandb.run.dir, 'model.p')
        with open(path, 'wb') as f:
            pickle.dump(model, f)


def qualitative_results(data, model, i=0, solver='dopri'):
    item = data.val[i]
    t = item['t']
    x = item['x']
    action = item['action']
    cm = plt.get_cmap('viridis')
    keypoints, keypoint_maps = model.encoder(x)
    norm = lambda x: (x - x.min()) / (x.max() - x.min())
    output_images = onp.concatenate([norm(keypoint_maps[..., i]) for i in range(keypoint_maps.shape[-1])], axis=2)
    heatmaps = (onp.concatenate([x, cm(output_images)[..., :3]], axis=-2) * 255).astype(onp.uint8)

    x_2 = x[:2, ...]
    keypoints, keypoint_maps = model.encoder(x_2)

    keypoints_pred = predict_constraint(model.constraint_fn, model.ode, t, keypoints, action, solver=solver)
    x_recon, gaussian_maps = model.renderer(keypoints_pred)

    prediction = (onp.concatenate([x, x_recon], axis=-2) * 255).astype(onp.uint8)
    return heatmaps, prediction


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--dynamics_weight', type=float, help='Wheight factor of dynamics loss.')
    return parser
