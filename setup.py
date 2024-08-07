from setuptools import find_packages, setup

setup(
    name='keycld',
    packages=find_packages(),
    install_requires=[
        'jax[cuda12]',
        'wandb',
        'jupyter',
        'matplotlib',
        'moviepy',
        'torch',
        'simple_parsing',
        'flax',
        'scikit-learn',
        'scikit-image',
        'pandas',
        'seaborn',
        'plotly',
        'dm_control @ git+https://github.com/rdaems/dm_control.git@b6b8717cd7b6c92304ee8da5b122ba4cb643d45c'
    ]
)
