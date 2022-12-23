import setuptools

__version__ = "1.3.0.post2"

URL = "https://pydgn.readthedocs.io/en/latest/"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pydgn",
    version=__version__,
    author="Federico Errica",
    author_email="f.errica@protonmail.com",
    description="A Python Package for Deep Graph Networks",
    long_description_content_type="text/markdown",
    url=URL,
    keywords=[
        "deep-graph-networks",
        "evaluation-framework",
        "deep-learning-for-graphs",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8,<3.10",
    install_requires=[
        "PyYAML >= 5.4",
        "requests >= 2.22.0",
        "tensorboard <= 2.11.0",
        "tqdm >= 4.47.0",
        "ogb <= 1.3.3, < 1.4.0",
        "protobuf == 3.20.3",
        "click <= 8.0.4",
        "ray == 2.1.0",
        "gpustat >= 1.0.0",
        "torch <= 1.13.0+*",
        "torch-geometric <= 2.1.0.post1",
        "torch-geometric-temporal <= 0.54.0",
        "wandb >= 0.12.15",
    ],
    packages=setuptools.find_packages(),
)
