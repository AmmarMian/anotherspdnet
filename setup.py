from setuptools import setup, find_packages

setup(
    name='anotherspdnet',
    author='Ammar Mian, Florent Bouchard, Guillaume Ginolhac, Paul Chauchat',
    author_email="ammar.mian@univ-smb.fr",
    version='0.1.0',
    packages=find_packages(include=['anotherspdnet', 'anotherspdnet.*']),
    python_requires='>=3.6',
    install_requires=[
        'torch>=2.0.1',
        'geoopt>=0.5.0',
        'tqdm'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pytest-sugar'
    ]
)
