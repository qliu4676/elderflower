from setuptools import setup, find_packages
from os import path

abspath = path.abspath(path.dirname(__file__))

with open(path.join(abspath, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(

    name='psffit', 

    version='0.1.0',  

    description='2D PSF Modeling', 

    url='https://github.com/NGC4676/PSF_Modeling', 

    author='Qing Liu',  

    author_email='qliu@astro.utoronto.ca',  

    keywords='astronomy PSF Bayesian fitting',

    package_dir={'': 'src'},

    packages=find_packages('src'),

    python_requires='>=3.5',

)
