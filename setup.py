#!/usr/bin/env python3

import site
import sys

from setuptools import find_packages, setup  # type: ignore

site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

setup(
    name="deskew",
    version="0.10.22",
    description="Skew detection and correction in images containing text",
    long_description="""Skew detection and correction in images containing text

Homepage: https://github.com/sbrunner/deskew""",
    license="MIT",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
    ],
    author="Stéphane Brunner",
    author_email="stephane.brunner@gmail.com",
    url="https://github.com/sbrunner/deskew",
    packages=find_packages(exclude=["tests.*"]),
    install_requires=["numpy", "scikit-image!=0.15.0"],
    entry_points={
        "console_scripts": [
            "deskew = deskew.cli:main",
        ],
    },
)
