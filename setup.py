from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='gpt-from-scratch',
    version='0.0.1',
    description='Decoder-only transformer model written from scratch',
    install_requires=requirements,
    license='MIT'
)