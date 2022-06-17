import wandb
from functools import partial
from simple_parsing import ArgumentParser

from keycld.data.dm import Data
from keycld.dm import ExperimentKeyCLD, validate


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--environment', type=str, help='Which DM control suite environment [pendulum, cartpole, acrobot].')
    parser.add_argument('--init_mode', type=str, help='State initialization mode [rest, random].')
    parser.add_argument('--control', type=str, help='Control mode [yes, no].')
    parser.add_arguments(ExperimentKeyCLD, dest='experiment')
    args = parser.parse_args()

    print(args)
    wandb.init(project=f'dm-{args.environment}-nc')
    wandb.config.update(args)
    data = Data(environment=args.environment, init_mode=args.init_mode, control=args.control)
    data.constraint_fn = None   # overwrite constraint_fn

    args.experiment.train(data, partial(validate, data))
