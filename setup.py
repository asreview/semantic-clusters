# based on https://github.com/pypa/sampleproject
# MIT License

# Always prefer setuptools over distutils
from setuptools import setup, find_namespace_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='asreview-semantic-clustering',
    description='Semantic clustering tool for the ASReview project',
    version='0.1',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/asreview/semantic-clusters',
    author='Utrecht University',
    author_email='asreview@uu.nl',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='asreview extension semantic clustering clusters visualization',
    packages=find_namespace_packages(include=['asreviewcontrib.*']),
    install_requires=[
        "numpy",
        "matplotlib",
        "asreview",
        "dash",
        "plotly",
        "sklearn",
        "transformers",
        "numpy",
        "seaborn",
        "torch",
    ],

    extras_require={
    },

    entry_points={
        "asreview.entry_points": [
            "semantic_clustering = asreviewcontrib.semantic_clustering.main:SemClusEntryPoint",  # noqa: E501
        ]
    },

    project_urls={
        'Bug Reports':
            "https://github.com/asreview/semantic-clusters/issues",
        'Source':
            "https://github.com/asreview/semantic-clusters",
    },
)
