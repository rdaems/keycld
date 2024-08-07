# KeyCLD: Learning Constrained Lagrangian Dynamics in Keypoint Coordinates from Images

### [Read the paper](https://arxiv.org/abs/2206.11030) | [View the project page](https://rdaems.github.io/keycld/)

https://user-images.githubusercontent.com/16489564/175288187-e52eb400-29db-4730-8845-8faffc8c0af6.mp4

KeyCLD learns Lagrangian dynamics from images. **(a)** An observation of a dynamical system is processed by a keypoint estimator model. **(b)** The model represents the positions of the keypoints with a set of spatial probability heatmaps. **(c)** Cartesian coordinates are extracted using spatial softmax and used as state representations to learn Lagrangian dynamics. **(d)** The information in the keypoint coordinates bottleneck suffices for a learned renderer model to reconstruct the original observation, including background, reflections and shadows. The keypoint estimator model, Lagrangian dynamics models and renderer model are jointly learned unsupervised on sequences of images.

## Installation

### KeyCLD

Clone this repository and install KeyCLD:
```
pip install .
```

A forked version of `dm_control` will be installed, with some changes to the environments to make it suitable for this work (see the paper Appendix for details).
This will also install mujoco.
If you have problems with (headless) rendering, see [https://github.com/google-deepmind/dm_control#rendering](https://github.com/google-deepmind/dm_control#rendering).

### Weights and Biases

KeyCLD uses [wandb](https://wandb.ai) for logging training and results.
If you are not familiar with wandb, follow the instructions when first running the code.

## Reproduce Experiments

Run the commands below to reproduce all experiments reported in the paper.
Check the results in wandb.
Or take a look in `notebooks/model.ipynb` to interact with the trained models (requires the model to finish training).

The data sets are generated automatically and cached at `/tmp/dm`.
The cache location can be changed at the top of `keycld/data/dm.py`.

### KeyCLD
```
python keycld/dm.py --environment=pendulum --init_mode=random --control=no --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python keycld/dm.py --environment=pendulum --init_mode=random --control=yes --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python keycld/dm.py --environment=cartpole --init_mode=random --control=no --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python keycld/dm.py --environment=cartpole --init_mode=random --control=underactuated --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python keycld/dm.py --environment=cartpole --init_mode=random --control=yes --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python keycld/dm.py --environment=acrobot --init_mode=random --control=no --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python keycld/dm.py --environment=acrobot --init_mode=random --control=underactuated --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python keycld/dm.py --environment=acrobot --init_mode=random --control=yes --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
```

### KeyCLD-NC
```
python ablations/no_constraint.py --environment=pendulum --init_mode=random --control=no --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python ablations/no_constraint.py --environment=pendulum --init_mode=random --control=yes --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python ablations/no_constraint.py --environment=cartpole --init_mode=random --control=no --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python ablations/no_constraint.py --environment=cartpole --init_mode=random --control=underactuated --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python ablations/no_constraint.py --environment=cartpole --init_mode=random --control=yes ---batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python ablations/no_constraint.py --environment=acrobot --init_mode=random --control=no ---batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python ablations/no_constraint.py --environment=acrobot --init_mode=random --control=underactuated ---batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python ablations/no_constraint.py --environment=acrobot --init_mode=random --control=yes ---batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
```

### NODE2
```
python ablations/node2.py --environment=pendulum --init_mode=random --control=no --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python ablations/node2.py --environment=pendulum --init_mode=random --control=yes --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python ablations/node2.py --environment=cartpole --init_mode=random --control=no --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python ablations/node2.py --environment=cartpole --init_mode=random --control=underactuated --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python ablations/node2.py --environment=cartpole --init_mode=random --control=yes --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python ablations/node2.py --environment=acrobot --init_mode=random --control=no --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python ablations/node2.py --environment=acrobot --init_mode=random --control=underactuated --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
python ablations/node2.py --environment=acrobot --init_mode=random --control=yes --batch_size=1 --dynamics_weight=1 --bce_weight=1 --solver=euler --learning_rate=0.0003 --num_epochs=40 --num_hidden_dim=32 --num_predicted_steps=4 && \
```

## Cite this research

To cite KeyCLD you can use the following bibtex:

```
@article{daems2024keycld,
  title={KeyCLD: Learning constrained Lagrangian dynamics in keypoint coordinates from images},
  author={Daems, Rembert and Taets, Jeroen and wyffels, Francis and Crevecoeur, Guillaume},
  journal={Neurocomputing},
  volume={573},
  pages={127175},
  year={2024},
  publisher={Elsevier}
}
```
