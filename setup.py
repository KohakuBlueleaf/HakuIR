from setuptools import setup, find_packages


setup(
    name="hakuir",
    packages=find_packages(),
    version="0.0.5",
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
