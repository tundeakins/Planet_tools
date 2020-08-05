import setuptools
import os, glob


here = os.path.abspath(os.path.dirname(__file__))

setuptools.setup(
    name="Planet_tools", # Replace with your own username
    version="0.0.2",
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
)
