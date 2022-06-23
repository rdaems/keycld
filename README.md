# KeyCLD: Learning Constrained Lagrangian Dynamics in Keypoint Coordinates from Images

### [Read the paper](https://arxiv.org/abs/2206.11030) | [View the project page](https://rdaems.github.io/keycld/)

https://user-images.githubusercontent.com/16489564/175272513-cc268678-e6a7-4ef9-a806-fd901e51e931.mp4

KeyCLD learns Lagrangian dynamics from images. **(a)** An observation of a dynamical system is processed by a keypoint estimator model. **(b)** The model represents the positions of the keypoints with a set of spatial probability heatmaps. **(c)** Cartesian coordinates are extracted using spatial softmax and used as state representations to learn Lagrangian dynamics. **(d)** The information in the keypoint coordinates bottleneck suffices for a learned renderer model to reconstruct the original observation, including background, reflections and shadows. The keypoint estimator model, Lagrangian dynamics models and renderer model are jointly learned unsupervised on sequences of images.

## Installation

### MuJoCo

Download [MuJoCo 2.2.0](https://github.com/deepmind/mujoco/releases/tag/2.2.0).
Unpack archive and place at `~/.mujoco/mujoco-2.2.0/`.

### KeyCLD

Install `jax==0.3.13` and `jaxlib==0.3.13` with cuda support: https://github.com/google/jax#pip-installation-gpu-cuda.
Optionally install JAX with CPU support (training will be very slow):
```
pip install jax==0.3.13 jaxlib==0.3.13
```

Clone this repository and install KeyCLD:
```
pip install .
```

### Weights and Biases

KeyCLD uses wandb.ai for logging training and results.
If you are not familiar with wandb, follow the instructions when first running the code.

## Reproduce Experiments

Run the commands below to reproduce all experiments reported in the paper.
Check the results in wandb.
Or take a look in `notebooks/model.ipynb` to interact with the trained models (requires the model to finish training).

### KeyCLD
```
python keycld/dm.py --environment pendulum --init_mode rest --control yes --batch_size=1 --dynamics_weight=0.0005 --learning_rate=0.0003 --num_epochs=100 --num_hidden_dim=32

python keycld/dm.py --environment cartpole --init_mode rest --control yes --batch_size=1 --dynamics_weight=0.0005 --learning_rate=0.0003 --num_epochs=100 --num_hidden_dim=32

python keycld/dm.py --environment acrobot --init_mode rest --control yes --batch_size=1 --dynamics_weight=0.0005 --learning_rate=0.0003 --num_epochs=100 --num_hidden_dim=32
```

### KeyCLD-NC
```
python ablations/no_constraint.py --environment pendulum --init_mode rest --control yes --batch_size=1 --dynamics_weight=0.0005 --learning_rate=0.0003 --num_epochs=100 --num_hidden_dim=32

python ablations/no_constraint.py --environment cartpole --init_mode rest --control yes --batch_size=1 --dynamics_weight=0.0005 --learning_rate=0.0003 --num_epochs=100 --num_hidden_dim=32

python ablations/no_constraint.py --environment acrobot --init_mode rest --control yes --batch_size=1 --dynamics_weight=0.0005 --learning_rate=0.0003 --num_epochs=100 --num_hidden_dim=32
```

### NODE2
```
python ablations/node2.py --environment pendulum --init_mode rest --control yes --batch_size=1 --dynamics_weight=0.0005 --learning_rate=0.0003 --num_epochs=100 --num_hidden_dim=32

python ablations/node2.py --environment cartpole --init_mode rest --control yes --batch_size=1 --dynamics_weight=0.0005 --learning_rate=0.0003 --num_epochs=100 --num_hidden_dim=32

python ablations/node2.py --environment acrobot --init_mode rest --control yes --batch_size=1 --dynamics_weight=0.0005 --learning_rate=0.0003 --num_epochs=100 --num_hidden_dim=32
```

### FC
```
python ablations/fc.py --environment pendulum --init_mode rest --control yes --batch_size=1 --dynamics_weight=0.0005 --learning_rate=0.0003 --num_epochs=100 --num_hidden_dim=32

python ablations/fc.py --environment cartpole --init_mode rest --control yes --batch_size=1 --dynamics_weight=0.0005 --learning_rate=0.0003 --num_epochs=100 --num_hidden_dim=32

python ablations/fc.py --environment acrobot --init_mode rest --control yes --batch_size=1 --dynamics_weight=0.0005 --learning_rate=0.0003 --num_epochs=100 --num_hidden_dim=32
```

## Cite this research

To cite KeyCLD you can use the following bibtex:

```
@article{daems2022keycld,
  title={KeyCLD: Learning Constrained Lagrangian Dynamics in Keypoint Coordinates from Images},
  author={Daems, Rembert and Taets, Jeroen and wyffels, Francis and Crevecoeur, Guillaume},
  journal={arXiv preprint arXiv:2206.11030},
  year={2022}
}
```
