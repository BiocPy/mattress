<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/mattress.svg?branch=main)](https://cirrus-ci.com/github/<USER>/mattress)
[![ReadTheDocs](https://readthedocs.org/projects/mattress/badge/?version=latest)](https://mattress.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/mattress/main.svg)](https://coveralls.io/r/<USER>/mattress)
[![PyPI-Server](https://img.shields.io/pypi/v/mattress.svg)](https://pypi.org/project/mattress/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/mattress.svg)](https://anaconda.org/conda-forge/mattress)
[![Monthly Downloads](https://pepy.tech/badge/mattress/month)](https://pepy.tech/project/mattress)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/mattress)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# mattress

This project provides [tatami matrix representations](https://github.com/tatami-inc) in Python.


## Install

Package is published to [PyPI](https://pypi.org/project/mattress/)

```shell
pip install mattress
```

## Usage

### Dense matrices
To convert a numpy dense matrix to tatami representation - 

```python
import numpy as np
from mattress import tatamize

x = np.random.rand(1000, 100)

tatamat = tatamize(y)
```

### Methods
Methods are available to access the matrix by `row`, `column`

```python
tatamat.row(0)
tatamat.column(1)
```

Also works if the matrix is row-major order.

```python
x = np.ones((2, 3), order='F')
tatamat = tatamize(y)
```

### Is your matrix sparse?

```python
from scipy.sparse import rand
m = rand(3, 4, density=0.25, format="csr", random_state=42)
tatamat = tatamize(m)
```

## Developer Notes


Steps to setup dependencies - 

- initialize git submodules in `extern/tatami` & `extern/tatami_hdf5`

First one needs to build the extern library, this would generate a shared object file to `src/mattress/core-[*].so`

```shell
python setup.py build_ext --inplace
```

For typical development workflows, run this for tests

```shell
python setup.py build_ext --inplace && tox
```

To rebuild the **ctypes** bindings with [the `wrap.py` helper](https://github.com/BiocPy/ctypes-wrapper)

```shell
wrap.py src/mattress/lib --py src/mattress/cpphelpers.py --cpp src/mattress/lib/bindings.cpp
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
