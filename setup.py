from setuptools import setup

setup(
    name='hakuir',
    packages=['hakuir'],
    version='0.0.1',
    install_requires=[
        'torch',
        'torchvision',
        'pillow',
        'numpy',
        'einops',
        'thop',
        'timm',
        'toml'
    ],
    entry_points={
        'console_scripts': [
            'hakuir = hakuir.console:console'
        ]
    }
)
