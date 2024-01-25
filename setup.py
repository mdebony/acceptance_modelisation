from setuptools import setup, find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='acceptance_modelisation',
    packages=find_packages(),
    version='0.2.0',
    description='Calculate 2D and 3d acceptance model for creating maps with gammapy (IACT analysis)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'gammapy>=1.1,<1.2',
        'numpy',
        'scipy',
        'astropy>=4.0',
        'regions>=0.7,<0.8'
    ],
    author='Mathieu de Bony de Lavergne',
    author_email='mathieu.debony@cea.fr'
)
