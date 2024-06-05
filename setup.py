from setuptools import setup

setup(
    name='partinet',
    version='0.1.0',
    packages=["partinet"],
    py_modules=['partinet'],
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'partinet = partinet:main',
        ],
    },
)