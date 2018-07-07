from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='empiricalutilities',
    version='0.0.9',
    description='A python project for empirical methods',
    author='Jason R Becker',
    author_email='jasonrichardbecker@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
      url='https://github.com/jason-r-becker/empiricalutilities',
    download_url='https://github.com/jason-r-becker/empiricalutilities/archive/0.1.tar.gz',
    keywords=['empirical latex OLS'],
      packages=find_packages(exclude=['contrib', 'docs', 'tests', 'build']),
    install_requires=['cycler', 'kiwisolver', 'matplotlib', 'mpmath', 'numpy', 'pandas',
                      'patsy', 'pprint', 'pyparsing', 'pytz', 'scipy', 'seaborn', 'six',
                      'statsmodels', 'sympy', 'tabulate']
)
