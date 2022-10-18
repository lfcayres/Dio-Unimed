from setuptools import setup, find_packages

with open("README.md", "r") as f:
    page_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="package-lfcayres",
    version="0.0.1",
    author="Luiz Felipe Cayres Vieira",
    author_email="lfcayres@gmail.com",
    description="Package Python Test",
    long_description=page_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lfcayres/dio-desafio-github--primeiro-repositorio",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
)