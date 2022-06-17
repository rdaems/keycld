import pickle
import os
import numpy as np
from tqdm import tqdm
import copy
from keycld import constraints


IMAGE_SIZE = 64
NUM_FRAMES = 50
NUM_RUNS = 500
DT = {
    'pendulum': 0.02,
    'cartpole': 0.01,
    'acrobot': 0.01,
}
NUM_KEYPOINTS = {
    'pendulum': 1,
    'cartpole': 2,
    'acrobot': 2,
}
CONSTRAINT_FNS = {
    'pendulum': constraints.pendulum_constraint_fn,
    'cartpole': constraints.cartpole_constraint_fn,
    'acrobot': constraints.acrobot_constraint_fn,
}
POSITION_KEYS = {
    'pendulum': 'orientation',
    'cartpole': 'position',
    'acrobot': 'orientations',
}


def generate_run(random_seed, environment, init_mode, control):
    from dm_control import suite
    assert environment in ['pendulum', 'cartpole', 'acrobot']
    assert init_mode in ['rest', 'random']
    assert control in ['yes', 'no']
    random_state = np.random.RandomState(random_seed)

    env = suite.load(environment, 'swingup', task_kwargs={'random': random_state})
    spec = env.action_spec()
    action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
    if control == 'no' or random_state.rand() < .2:
        action = action * 0
    time_step = env.reset()

    if init_mode == 'random':
        if environment == 'pendulum':
            q = random_state.uniform(- np.pi, np.pi)
            q_dot = random_state.uniform(-.1, .1)
        elif environment == 'cartpole':
            q = random_state.uniform([-1, - np.pi], [1, np.pi])
            q_dot = random_state.uniform([-.1, -.1], [.1, .1])
        elif environment == 'acrobot':
            q = random_state.uniform([- np.pi, - np.pi], [np.pi, np.pi])
            q_dot = random_state.uniform([-.1, -.1], [.1, .1])
        with env.physics.reset_context():
            env.physics.data.qpos[:] = q
            env.physics.data.qvel[:] = q_dot

    position_key = POSITION_KEYS[environment]
    frames, ticks, positions, velocities = [], [], [], []
    for step in range(NUM_FRAMES):
        time_step = env.step(action)
        frames.append(env.physics.render(camera_id=0, height=IMAGE_SIZE, width=IMAGE_SIZE))
        ticks.append(env.physics.data.time)
        observation = copy.deepcopy(time_step.observation)
        positions.append(observation[position_key])
        velocities.append(observation['velocity'])

    return {
        't': np.array(ticks).astype(np.float32),
        'x': np.array(frames).astype(np.float32) / 255,
        'action': np.array(action).astype(np.float32),
        'positions': np.array(positions).astype(np.float32),
        'velocities': np.array(velocities).astype(np.float32),
    }


def generate_grid(environment, num_grid_points=16):
    from dm_control import suite
    """Generate grid over state space, usefull for e.g. plotting potential energy."""
    assert environment in ['pendulum', 'cartpole', 'acrobot']
    env = suite.load(environment, 'swingup')
    if environment == 'pendulum':
        qs = np.linspace(- np.pi, np.pi, num_grid_points)[np.newaxis, :]
    elif environment == 'cartpole':
        qs = np.meshgrid(np.linspace(-1, 1, num_grid_points), np.linspace(-np.pi, np.pi, num_grid_points))
        qs = np.stack(qs, axis=-1)
    elif environment == 'acrobot':
        qs = np.meshgrid(np.linspace(-np.pi, np.pi, num_grid_points), np.linspace(-np.pi, np.pi, num_grid_points))
        qs = np.stack(qs, axis=-1)

    images = []
    for q_list in qs:
        image_list = []
        for q in q_list:
            with env.physics.reset_context():
                env.physics.data.qpos[:] = q
            image = env.physics.render(camera_id=0, height=IMAGE_SIZE, width=IMAGE_SIZE)
            image_list.append(image)
        images.append(image_list)
    return {
        'positions': qs.astype(np.float32),
        'x': np.array(images).astype(np.float32) / 255,
    }


def generate_runs(environment, init_mode, control, num_runs=NUM_RUNS):
    print(f'Generating runs for environment {environment} with init mode {init_mode} and control {control}.')
    runs = [generate_run(random_seed, environment, init_mode, control) for random_seed in tqdm(range(num_runs))]
    # runs = Parallel(n_jobs=16)(delayed(generate_run)(random_seed, environment, init_mode, control) for random_seed in range(num_runs))
    return runs


class Data:
    def __init__(self, environment, init_mode, control, overwrite_cache=False):
        cache_path = f'/tmp/keycld_cache_{environment}_{init_mode}_{control}.p'
        if os.path.exists(cache_path) and not overwrite_cache:
            print(f'Loading {cache_path}')
            with open(cache_path, 'rb') as f:
                runs, self.grid = pickle.load(f)
        else:
            runs = generate_runs(environment, init_mode, control)
            self.grid = grid = generate_grid(environment)
            with open(cache_path, 'wb') as f:
                pickle.dump((runs, grid), f)
        self.environment = environment
        self.dt = DT[environment]
        self.num_timesteps = NUM_FRAMES
        self.n = NUM_KEYPOINTS[environment]
        self.train = runs[:-50]
        self.val = runs[-50:]
        self.constraint_fn = CONSTRAINT_FNS[environment]
        self.image_size = IMAGE_SIZE
