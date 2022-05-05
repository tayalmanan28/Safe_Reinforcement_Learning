from envs.unicycle_env import UnicycleEnv
from envs.simulated_cars_env import SimulatedCarsEnv

"""
This file includes a function that simply returns one of the two supported environments. 
"""

def build_env(args):
    """Build our custom gym environment."""

    if args.env_name == 'Unicycle':
        return UnicycleEnv()
    elif args.env_name == 'SimulatedCars':
        return SimulatedCarsEnv()
    else:
        raise Exception('Env {} not supported!'.format(args.env_name))
