import setuptools

__version__ = "1.0.9"

URL = 'https://pydgn.readthedocs.io/en/latest/'

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
        'deep-graph-networks',
        'evaluation-framework',
        'deep-learning-for-graphs'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8,<3.10",
    install_requires=[
        'PyYAML >= 5.4',
        'networkx >= 2.3',
        'requests >= 2.22.0',
        'matplotlib >= 3.3.4',
        'seaborn >= 0.9.0',
        'tensorboard >= 2.1.1',
        'tqdm >= 4.47.0',
        'ogb >= 1.3.3',
        'aioredis >= 1.3.1',
        'ray >= 1.5.2',
        'gpustat >= 0.6.0',
        'torch >= 1.10.2',
        'torch-geometric >= 2.0.3',
    ],
    packages=setuptools.find_packages()
)
