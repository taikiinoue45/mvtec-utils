from typing import List

from setuptools import find_packages, setup


def get_install_requires() -> List[str]:
    return open("requirements.txt").read().splitlines()


def get_readme() -> str:
    return open("README.md").read()


setup(
    name="mvtec",
    version="0.0.0",
    description="Toolbox for Unsupervised Anomaly Detection on MVTec AD",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    author="taikiinoue45",
    author_email="taikiinoue45@gmail.com",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=get_install_requires(),
    url="https://github.com/taikiinoue45/mvtec-utils",
    classifiers=["Programming Language :: Python :: 3.6"],
)
