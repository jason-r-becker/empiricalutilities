from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst')) as f:
    long_description = f.read()

setup(
    name='empiricalutilities',
    version='0.1.8',
    description='A Python project for empirical data manipulation.',
    long_description=long_description,
    author='Jason R Becker',
    author_email='jasonrichardbecker@gmail.com',
    python_requires='>=3.6.0',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        ],
    url='https://github.com/jason-r-becker/empiricalutilities',
    download_url='https://github.com/jason-r-becker/empiricalutilities/archive/0.1.tar.gz',
    keywords='empirical LaTeX OLS'.split(),
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'build']),
    install_requires=[
        'cycler',
        'kiwisolver',
        'matplotlib',
        'mpmath',
        'numpy',
        'pandas',
        'patsy',
        'pprint',
        'pyparsing',
        'pytz',
        'scipy',
        'seaborn',
        'six',
        'statsmodels',
        'sympy',
        'tabulate',
        ]
)
