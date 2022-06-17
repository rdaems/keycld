from setuptools import find_packages, setup

setup(
    name='keycld',
    packages=find_packages(),
    install_requires=[
        'wandb==0.12.18',
        'jupyter==1.0.0',
        'matplotlib==3.5.2',
        'moviepy==1.0.3',
        'torch==1.11.0',
        'simple_parsing==0.0.20',
        'flax==0.5.1',
        'scikit-learn==1.1.1',
        'scikit-image==0.19.3',
        'pandas==1.4.2',
        'dm_control @ git+ssh://git@github.com/rdaems/dm_control.git@c682e626fde95a98b53f67f07b0c1021e4200bb8'
    ]
)