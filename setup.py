import os
from setuptools import setup, find_packages

abspath = os.path.abspath(os.path.dirname(__file__))

def readme():
    with open('README.md') as f:
        return f.read()

install_requires = []

#requirementPath = os.path.join(abspath, 'requirements.txt')

#if os.path.isfile(requirementPath):
#    with open(requirementPath) as f:
#        install_requires = f.read().splitlines()


setup(

    name='elderflower', 

    version='0.3',  

    description='Wide-angle PSF modeling for low surface brightness imaging', 

    long_description=readme(),
    
    long_description_content_type='text/markdown',

    url='https://github.com/NGC4676/elderflower', 

    author='Qing Liu',  

    author_email='qliu@astro.utoronto.ca',  

    keywords='astronomy PSF fitting LSB',

    packages=find_packages(include=['elderflower','elderflower.']),

    python_requires='>=3.5',

    install_requires=install_requires,

)
