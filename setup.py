import os
from setuptools import setup, find_packages

abspath = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(abspath, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = []

#requirementPath = os.path.join(abspath, 'requirements.txt')

#if os.path.isfile(requirementPath):
#    with open(requirementPath) as f:
#        install_requires = f.read().splitlines()


setup(

    name='elderflower', 

    version='0.1.0',  

    description='Wide-angle PSF modeling for low surface brightness imaging', 

    long_description=long_description,

    url='https://github.com/NGC4676/elderflower', 

    author='Qing Liu',  

    author_email='qliu@astro.utoronto.ca',  

    keywords='astronomy PSF fitting LSB',

    packages=find_packages(),

    python_requires='>=3.5',

    install_requires=install_requires,

)
