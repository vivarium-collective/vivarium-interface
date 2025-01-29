import re
from setuptools import setup, find_packages


VERSION = '0.0.1'


with open("README.md", "r") as readme:
    description = readme.read()
    # Patch the relative links to absolute URLs that will work on PyPI.
    description2 = re.sub(
        r']\(([\w/.-]+\.png)\)',
        r'](https://github.com/vivarium-collective/vivarium-interface/raw/main/\1)',
        description)
    long_description = re.sub(
        r']\(([\w/.-]+)\)',
        r'](https://github.com/vivarium-collective/vivarium-interface/blob/main/\1)',
        description2)

setup(
    name="vivarium-interface",
    version=VERSION,
    author="Ryan Spangler, Eran Agmon",
    author_email="ryan.spangler@gmail.com, agmon.eran@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vivarium-collective/vivarium-interface",
    # packages=find_packages(),
    packages=[
        'vivarium',
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.6",
    install_requires=[
        "process-bigraph>=0.0.25",
        "bigraph-viz",
    ]
)
