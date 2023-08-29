from numpy import ndarray, float64
from . import cpphelpers as lib

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
        """Compute row sums.

        Args:
            n (int, optional): Number of threads.

        Returns:
            ndarray: Array of row sums.
        """
        output = ndarray((self.nrow(),), dtype=float64)
        lib.compute_row_sums(self.ptr, output.ctypes.data, num_threads)
        return output

    def column_sums(self, num_threads: int = 1) -> ndarray:
        """Compute column sums.

        Args:
            n (int, optional): Number of threads.

        Returns:
            ndarray: Array of column sums.
        """
        output = ndarray((self.ncol(),), dtype=float64)
        lib.compute_column_sums(self.ptr, output.ctypes.data, num_threads)
        return output
