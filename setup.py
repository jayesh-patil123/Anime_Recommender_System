from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ml_project_02",
    version="0.1",
    author="Jayesh",
    packages=find_packages(),
    install_requires = requirements,
)