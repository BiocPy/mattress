import os
from typing import List
import assorthead
import inspect

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def includes() -> List[str]:
    """Provides access to C++ headers (including tatami) for downstream packages.

    Returns:
        List[str]: List of paths to the header files.
    """
    dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return [
        assorthead.includes(),
        os.path.join(dirname, "include"),
    ]
