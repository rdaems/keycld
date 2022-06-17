import jax.numpy as jnp


def horizontal_constraint_fn(state, permutation=[0]):
    i = permutation[0]
    x, y = state[::2], state[1::2]
    c = y[i]
    return jnp.array([c])


def distance_constraint_fn(state, permutation=[0, 1]):
    i, j = permutation[:2]
    x, y = state[::2], state[1::2]
    c = jnp.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) - 0.5
    return jnp.array([c])


def pendulum_constraint_fn(state, permutation=[0]):
    i = permutation[0]
    x, y = state[::2], state[1::2]
    c = jnp.sqrt(x[i] ** 2 + y[i] ** 2) - 0.5
    return jnp.array([c])


def cartpole_constraint_fn(state, permutation=[0, 1]):
    i, j = permutation[:2]
    x, y = state[::2], state[1::2]
    return jnp.array([
        y[i],
        jnp.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) - 0.5,
    ])


def acrobot_constraint_fn(state, permutation=[0, 1]):
    i, j = permutation[:2]
    x, y = state[::2], state[1::2]
    return jnp.array([
        jnp.sqrt(x[i] ** 2 + y[i] ** 2) - 0.3,
        jnp.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) - 0.3,
    ])
