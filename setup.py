from setuptools import setup

setup(
    name="hakuir",
    packages=["hakuir"],
    version="0.0.2",
    install_requires=[
        "torch",
        "torchvision",
        "pillow",
        "numpy",
        "einops",
        "thop",
        "timm",
        "toml",
    ],
    entry_points={"console_scripts": ["hakuir = hakuir.cli:cli"]},
)
