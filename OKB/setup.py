from setuptools import find_packages, setup

REQUIRED = ['numpy>=1.21,<2',
            'orbax-checkpoint==0.4.8',
            'torch',
            # flax==0.7.5's configurations.py calls jax.config.define_bool_state, removed in
            # later jax; pin to what flax 0.7.5 was actually built against.
            'jax==0.4.20',
            'jaxlib==0.4.20',
            'optax==0.1.7',
            'chex==0.1.85',
            'flax==0.7.5',
            'cvxpylayers',
            'mo-gymnasium==1.3.1',
            'cvxpy',
            'rliable',
            'wandb',
            'pymoo>=0.6.1.3',
            'seaborn',
            'pandas',
            'fire',
            'scipy==1.12.0',
            'pycddlib==2.1.6',
            'tensorboard',       # required by rl.rl_algorithm (torch.utils.tensorboard)
            'opencv-python',     # required by rl.envs.item_collector
            'highway-env',       # required by rl.envs.highway
            'mujoco==3.1.6',     # pin to a version with a prebuilt wheel; latest can fail
                                 # to build from source (missing MUJOCO_PATH)
            'gymnasium-robotics', # required by rl.envs.mo_push (PickPlace)
            ]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='option-keyboard-basis',
    version='0.1',
    packages=['rl',],
    install_requires=REQUIRED,
    long_description=long_description,
    description='Code for the paper Constructing an Optimal Behavior Basis for the Option Keyboard.'
)
