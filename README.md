<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/mattress.svg?branch=main)](https://cirrus-ci.com/github/<USER>/mattress)
[![ReadTheDocs](https://readthedocs.org/projects/mattress/badge/?version=latest)](https://mattress.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/mattress/main.svg)](https://coveralls.io/r/<USER>/mattress)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/mattress.svg)](https://anaconda.org/conda-forge/mattress)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/mattress)
-->

[![PyPI-Server](https://img.shields.io/pypi/v/mattress.svg)](https://pypi.org/project/mattress/)
[![Monthly Downloads](https://static.pepy.tech/badge/mattress/month)](https://pepy.tech/project/mattress)
![Unit tests](https://github.com/tatami-inc/mattress/actions/workflows/pypi-test.yml/badge.svg)

# Python bindings for tatami

## Overview

The **mattress** package implements Python bindings to the [**tatami**](https://github.com/tatami-inc) C++ library for matrix representations.
Downstream packages can use **mattress** to develop C++ extensions that are interoperable with many different matrix classes, e.g., dense, sparse, delayed or file-backed.
**mattress** is inspired by the [**beachmat**](https://bioconductor/packages/beachmat) Bioconductor package, which does the same thing for R packages.

## Instructions

**mattress** is published to [PyPI](https://pypi.org/project/mattress/), so installation is simple:

```shell
pip install mattress
```

**mattress** is intended for Python package developers writing C++ extensions that operate on matrices.

1. Add `assorthead.includes()` to the `include_dirs=` of your `Extension()` definition in `setup.py`.
This will give us access to the various **tatami** headers to compile your C++ code.
2. Call `mattress.tatamize()` on a Python matrix object to wrap it in a **tatami**-compatible C++ representation. 
This returns a `TatamiMatrixPointer` with a `ptr` property that contains a pointer to the C++ matrix.
3. Pass the `ptr` to **pybind11**-wrapped C++ code as a `std::shared_ptr<tatami::Matrix<double, uint32_t> >`.

So, for example, the C++ code in our downstream package might look like this:

```cpp
int do_something(const std::shared_ptr<tatami::Matrix<double, uint32_t> >& mat) {
    // Do something with the tatami interface.
    return 1;
}

PYBIND11_MODULE(lib_downstream, m) {
    m.def("do_something", &do_something);
}
```

Which can then be called from Python:

```python
from . import lib_downstream as lib
from mattress import tatamize

def do_something(x):
    tmat = tatamize(x)
    return lib.do_something(tmat.ptr)
```

## Supported matrices

Dense **numpy** matrices of varying numeric type:

```python
import numpy as np
from mattress import tatamize
x = np.random.rand(1000, 100)
tatamat = tatamize(x)

ix = (x * 100).astype(np.uint16)
tatamat2 = tatamize(ix)
```

Compressed sparse matrices from **scipy** with varying index/data types:

```python
from scipy import sparse as sp
from mattress import tatamize

xc = sp.random(100, 20, format="csc")
tatamat = tatamize(xc)

xr = sp.random(100, 20, format="csc", dtype=np.uint8)
tatamat2 = tatamize(xr)
```

Delayed arrays from the [**delayedarray**](https://github.com/BiocPy/DelayedArray) package:

```python
from delayedarray import DelayedArray
from scipy import sparse as sp
from mattress import tatamize
import numpy

xd = DelayedArray(sp.random(100, 20, format="csc"))
xd = numpy.log1p(xd * 5)

tatada = tatamize(xd)
```

To be added:

- File-backed matrices from the [**FileBackedArray**](https://github.com/BiocPy/FileBackedArray) package, including HDF5 and TileDB.
- Arbitrary Python matrices?

## Utility methods

The `TatamiNumericPointer` instance returned by `tatamize()` provides a few Python-visible methods for querying the C++ matrix.

```python
tatamat.nrow() // number of rows
tatamat.column(1) // contents of column 1
tatamat.sparse() // whether the matrix is sparse.
```

It also has a few methods for computing common statistics:

```python
tatamat.row_sums()
tatamat.column_variances(num_threads = 2)

grouping = [i%3 for i in range(tatamat.ncol())]
tatamat.row_medians_by_group(grouping)

tatamat.row_nan_counts()
tatamat.column_ranges()
```

These are mostly intended for non-intensive work or testing/debugging.
It is expected that any serious computation should be performed by iterating over the matrix in C++.

## Operating on an existing pointer

If we already have a `TatamiNumericPointer`, we can easily apply additional operations by wrapping it in the relevant **delayedarray** layers and calling `tatamize()` afterwards.
For example, if we want to add a scalar, we might do:

```python
from delayedarray import DelayedArray
from mattress import tatamize
import numpy

x = numpy.random.rand(1000, 10)
tatamat = tatamize(x)

wrapped = DelayedArray(tatamat) + 1
tatamat2 = tatamize(wrapped)
```

This avoids relying on `x` and is more efficient as it re-uses the `TatamiNumericPointer` generated from `x`.

## Extending `tatamize()`

Developers of downstream packages can extend **mattress** to custom matrix classes by registering the relevant methods with the `tatamize()` generic.
This should return a `TatamiNumericPointer` object containing a shared pointer to a `tatami::Matrix<double, uint32_t>` instance.
Once this is done, all calls to `tatamize()` will be able to handle matrices of the registered type.

```python
from . import lib_downstream as lib
import mattress

@mattress.tatamize.register
def _tatamize_my_custom_matrix(x: MyCustomMatrix):
    data = x.some_internal_data
    return mattress.TatamiNumericPointer(lib.initialize_custom(data), obj=[data])
```

If the initialized `tatami::Matrix` contains references to Python-managed data, e.g., in NumPy arrays,
we must ensure that the data is not garbage-collected during the lifetime of the `tatami::Matrix`.
This is achieved by storing a reference to the data in the `obj=` argument of the `TatamiNumericPointer`.
