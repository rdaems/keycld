import os
import pickle
import jax
import jax.numpy as jnp
from tqdm import tqdm
import optax
import numpy as onp
import matplotlib.pyplot as plt
from functools import partial
import wandb
from dataclasses import dataclass

from keycld.losses import loss_fn_step
from keycld.util import reduce_mean, NumpyLoader
from keycld.models import predict_run

# error on nan
jax.config.update('jax_debug_nans', True)


@dataclass
class ExperimentBase:
    num_epochs: int             # Number of training epochs.
    learning_rate: float        # Learning rate.
    batch_size: int             # Batch size.
    num_hidden_dim: int         # Number of hidden layers in models.
    num_predicted_steps: int    # Number of predicted steps in dynamics loss.
    dynamics_weight: float      # Weight factor of dynamics loss.

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
        dataloader = NumpyLoader(data.train, batch_size=self.batch_size, num_workers=1, shuffle=True)

        model = self.construct_model(data)
        params = model.init(jax.random.PRNGKey(1))
        self.configure_optimizers(params)
        loss_grad_fn = jax.jit(jax.value_and_grad(reduce_mean(jax.vmap(partial(loss_fn_step, self.dynamics_weight, self.num_predicted_steps, model), in_axes=(None, 0, 0))), has_aux=True))

        for epoch in range(self.num_epochs):
            total_loss, total_loss_aux = [], []
            with tqdm(total=len(dataloader)) as pbar:
                for batch in dataloader:
                    augmentation_permutations = onp.random.randint(0, 8, (self.batch_size, data.num_timesteps))
                    (loss_val, loss_aux), grads = loss_grad_fn(params, batch, augmentation_permutations)
                    if jnp.isnan(loss_val):
                        raise ValueError('NaN detected!')
                    params = self.update(params, grads)
                    total_loss.append(loss_val)
                    total_loss_aux.append(loss_aux)

                    pbar.set_description(f'[Epoch {epoch}] Loss: {loss_val:.04f}, ' + ', '.join([f'{key}: {value:.04f}' for key, value in loss_aux.items()]))
                    pbar.update(1)
                total_loss = onp.mean(total_loss)
                total_loss_aux = {key: onp.mean([d[key] for d in total_loss_aux]) for key in total_loss_aux[0]}
                wandb.log({'loss': total_loss}, step=epoch)
                wandb.log(total_loss_aux, step=epoch)
                pbar.set_description(f'[Epoch {epoch}] Loss: {total_loss:.04f}, ' + ', '.join([f'{key}: {value:.04f}' for key, value in total_loss_aux.items()]))

            validate_fn(data, model, params, epoch)
            self.save_params(params, epoch)

    def save_params(self, params, epoch):
        path = os.path.join(wandb.run.dir, f'params_{epoch}.p')
        with open(path, 'wb') as f:
            pickle.dump(params, f)

