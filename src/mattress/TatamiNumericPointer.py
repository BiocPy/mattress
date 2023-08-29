from numpy import ndarray, float64
from . import cpphelpers as lib
from typing import Tuple

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


class TatamiNumericPointer:
    """Initialize a Tatami Numeric Ponter object.

    Attributes:
        ptr (int): Pointer address to a Mattress instance wrapping a tatami matrix. This can be passed as
            a ``void *`` to C++ code and then cast to a ``Mattress *`` for actual use.

        obj (list): List of Python objects referenced by the tatami instance.
            This is stored here to avoid garbage collection.
    """

    def __init__(self, ptr: int, obj: list):
        self.ptr = ptr
        self.obj = obj

    def __del__(self):
        lib.free_mat(self.ptr)

    def nrow(self) -> int:
        """Get number of rows.

        Returns:
            int: Number of rows.
        """
        return lib.extract_nrow(self.ptr)

    def ncol(self) -> int:
        """Get number of columns.

        Returns:
            int: Number of columns.
        """
        return lib.extract_ncol(self.ptr)

    def sparse(self) -> bool:
        """Is the matrix sparse?

        Returns:
            bool: True if matrix is sparse.
        """
        return lib.extract_sparse(self.ptr) > 0

    def row(self, r: int) -> ndarray:
        """Access a row from the tatami matrix.

        Args:
            r (int): Row to access.

        Returns:
            ndarray: Row from the matrix. This is always in double-precision,
            regardless of the underlying representation.
        """
        output = ndarray((self.ncol(),), dtype="float64")
        lib.extract_row(self.ptr, r, output.ctypes.data)
        return output

    def column(self, c: int) -> ndarray:
        """Access a column from the tatami matrix.

        Args:
            c (int): Column to access.

        Returns:
            ndarray: Column from the matrix. This is always in double-precisino,
            regardless of the underlying representation.
        """
        output = ndarray((self.nrow(),), dtype="float64")
        lib.extract_column(self.ptr, c, output.ctypes.data)
        return output

    def row_sums(self, num_threads: int = 1) -> ndarray:
        """Convenience method to compute row sums.

        Args:
            num_threads (int, optional): Number of threads.

        Returns:
            ndarray: Array of row sums.
        """
        output = ndarray((self.nrow(),), dtype=float64)
        lib.compute_row_sums(self.ptr, output.ctypes.data, num_threads)
        return output

    def column_sums(self, num_threads: int = 1) -> ndarray:
        """Convenience method to compute column sums.

        Args:
            num_threads (int, optional): Number of threads.

        Returns:
            ndarray: Array of column sums.
        """
        output = ndarray((self.ncol(),), dtype=float64)
        lib.compute_column_sums(self.ptr, output.ctypes.data, num_threads)
        return output

    def row_variances(self, num_threads: int = 1) -> ndarray:
        """Convenience method to compute row variances.

        Args:
            num_threads (int, optional): Number of threads.

        Returns:
            ndarray: Array of row variances.
        """
        output = ndarray((self.nrow(),), dtype=float64)
        lib.compute_row_variances(self.ptr, output.ctypes.data, num_threads)
        return output

    def column_variances(self, num_threads: int = 1) -> ndarray:
        """Convenience method to compute column variances.

        Args:
            num_threads (int, optional): Number of threads.

        Returns:
            ndarray: Array of column variances.
        """
        output = ndarray((self.ncol(),), dtype=float64)
        lib.compute_column_variances(self.ptr, output.ctypes.data, num_threads)
        return output

    def row_medians(self, num_threads: int = 1) -> ndarray:
        """Convenience method to compute row medians.

        Args:
            num_threads (int, optional): Number of threads.

        Returns:
            ndarray: Array of row medians.
        """
        output = ndarray((self.nrow(),), dtype=float64)
        lib.compute_row_medians(self.ptr, output.ctypes.data, num_threads)
        return output

    def column_medians(self, num_threads: int = 1) -> ndarray:
        """Convenience method to compute column medians.

        Args:
            num_threads (int, optional): Number of threads.

        Returns:
            ndarray: Array of column medians.
        """
        output = ndarray((self.ncol(),), dtype=float64)
        lib.compute_column_medians(self.ptr, output.ctypes.data, num_threads)
        return output

    def row_mins(self, num_threads: int = 1) -> ndarray:
        """Convenience method to compute row minima.

        Args:
            num_threads (int, optional): Number of threads.

        Returns:
            ndarray: Array of row minima.
        """
        output = ndarray((self.nrow(),), dtype=float64)
        lib.compute_row_mins(self.ptr, output.ctypes.data, num_threads)
        return output

    def column_mins(self, num_threads: int = 1) -> ndarray:
        """Convenience method to compute column minima.

        Args:
            num_threads (int, optional): Number of threads.

        Returns:
            ndarray: Array of column mins.
        """
        output = ndarray((self.ncol(),), dtype=float64)
        lib.compute_column_mins(self.ptr, output.ctypes.data, num_threads)
        return output

    def row_maxs(self, num_threads: int = 1) -> ndarray:
        """Convenience method to compute row maxima.

        Args:
            num_threads (int, optional): Number of threads.

        Returns:
            ndarray: Array of row maxima.
        """
        output = ndarray((self.nrow(),), dtype=float64)
        lib.compute_row_maxs(self.ptr, output.ctypes.data, num_threads)
        return output

    def column_maxs(self, num_threads: int = 1) -> ndarray:
        """Convenience method to compute column maxima.

        Args:
            num_threads (int, optional): Number of threads.

        Returns:
            ndarray: Array of column maxs.
        """
        output = ndarray((self.ncol(),), dtype=float64)
        lib.compute_column_maxs(self.ptr, output.ctypes.data, num_threads)
        return output

    def row_ranges(self, num_threads: int = 1) -> Tuple[ndarray, ndarray]:
        """Convenience method to compute row ranges.

        Args:
            num_threads (int, optional): Number of threads.

        Returns:
            Tuple[ndarray, ndarray]: Tuple containing the row minima and maxima.
        """
        min_output = ndarray((self.nrow(),), dtype=float64)
        max_output = ndarray((self.nrow(),), dtype=float64)
        lib.compute_row_ranges(
            self.ptr, min_output.ctypes.data, max_output.ctypes.data, num_threads
        )
        return (min_output, max_output)

    def column_ranges(self, num_threads: int = 1) -> Tuple[ndarray, ndarray]:
        """Convenience method to compute column ranges.

        Args:
            num_threads (int, optional): Number of threads.

        Returns:
            Tuple[ndarray, ndarray]: Tuple containing the column minima and maxima.
        """
        min_output = ndarray((self.ncol(),), dtype=float64)
        max_output = ndarray((self.ncol(),), dtype=float64)
        lib.compute_column_ranges(
            self.ptr, min_output.ctypes.data, max_output.ctypes.data, num_threads
        )
        return (min_output, max_output)
