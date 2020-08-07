import setuptools
import os, glob
from version import __version__

here = os.path.abspath(os.path.dirname(__file__))

setuptools.setup(
    name="Planet_tools", # Replace with your own username
    version=__version__,
    author="Babatunde Akinsanmi",
    author_email="tunde.akinsanmi@astro.up.pt",
    description="Package with useful functions for exoplanetary research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tundeakins/Planet_tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    extras_require = { "dev": [ "pytest >=3.7 "],},
)
