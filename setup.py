import os
from setuptools import setup, find_packages

abspath = os.path.abspath(path.dirname(__file__))

with open(os.path.join(abspath, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = []
if os.path.isfile(requirementPath):
    with open(os.path.join(abspath, 'requirements.txt')) as f:
        install_requires = f.read().splitlines()


setup(

    name='psffit', 

    version='0.1.0',  

    description='2D PSF Modeling', 

    long_description=long_description,

    url='https://github.com/NGC4676/PSF_Modeling', 

    author='Qing Liu',  

    author_email='qliu@astro.utoronto.ca',  

    keywords='astronomy PSF Bayesian fitting',

    packages=find_packages("psffit"),

    python_requires='>=3.5',

    install_requires=install_requires,

)
