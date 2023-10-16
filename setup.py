from setuptools import setup, find_packages

setup(
    name='anotherspdnet',
    author='Ammar Mian, Florent Bouchard',
    author_email="ammar.mian@univ-smb.fr",
    version='0.1.0',
    packages=find_packages(include=['anotherspdnet', 'anotherspdnet.*']),
    install_requires=[
        'torch>=2.0.1',
        'geomstats>=2.0.0', 
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pytest-sugar'
    ]
)
