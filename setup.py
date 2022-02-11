from setuptools import setup
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='deep-reinforcement-learning',
    version='0.1.1',
    description='Deep Reinforcement Learning repository',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dongyoon-wee/deep-reinforcement-learning',
    author='Dongyoon Wee',
    author_email='dongyoon.wee@gmail.com',

    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6'
    ],

    install_requires=[
        'tensorflow>=1.7,<1.8',
        'Pillow>=4.2.1',
        'matplotlib',
        'numpy>=1.13.3,<=1.14.5',
        'jupyter',
        'pytest>=3.2.2',
        'docopt',
        'pyyaml',
        'protobuf>=3.6,<3.16',
        'grpcio>=1.11.0,<1.12.0'],

    python_requires=">=3.6,<3.7",
)