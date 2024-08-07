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
        'dm_control @ git+https://github.com/rdaems/dm_control.git@e2153e8763f8765f7f5b0c1e342987a531bead0a'
    ]
)
