from setuptools import setup, find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='acceptance_modelisation',
    packages=find_packages(),
    version='0.1.1-dev',
    description='Calculate 2D acceptance model for creating maps with gammapy (IACT analysis)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'gammapy>=0.19,<0.20',
        'numpy',
        'astropy>=4.0',
        'regions>=0.5,<0.6'
    ],
    author='Mathieu de Bony de Lavergne',
    author_email='lavergne@lapp.in2p3.fr'
)
