import io
import os

from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('rageval/version.py').read())

short_description = 'Evaluation tools for Retrieval-augmented Generation (RAG) methods.'

# Get the long description from the README file
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = [
    'numpy >= 1.14',
    'tqdm >= 4.23.4',
    'hyperopt >= 0.1.1',
    'h5py >= 2.8.0',
    'openai == 1.10.0',
    'datasets == 2.16.1',
    'langchain == 0.1.4',
    'transformers == 4.37.2',
    'torch == 2.2.0',
    'pandas == 2.0.0',
    'nltk == 3.8.1',
    'spacy == 3.7.4',
    'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl',
    'rouge_score == 0.1.2',
    'sacrebleu == 2.3.3'
]

extras_requires = {
    'tests': [
        'coverage >= 4.3.4',
        'codecov >= 2.0.15',
        'pytest >= 3.7.4',
        'pytest-cov >= 2.4.0',
        'flake8 == 7.0.0',
        'pydocstyle == 2.1',
        'flake8_docstrings >= 1.7.0'
    ],
    'benchmarks': [
        'accelerate == 0.27.2',
        'sentencepiece == 0.2.0',
        'protobuf == 4.25.3'
    ]
}


setup(
    name="RagEval",
    version=__version__,
    author="Wenshan Wang, Yixing Fan, etc.",
    author_email="wangwenshan@ict.ac.cn",
    description=short_description,
    license="Apache 2.0",
    keywords="RAG evaluation tools",
    url="https://github.com/gomate-community/rageval",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3.6'
    ],
    install_requires=install_requires,
    extras_require=extras_requires
)
