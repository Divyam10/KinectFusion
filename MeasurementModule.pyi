from typing import Callable
import numpy as np
# Numpy needs to be imported like this: import numpy as np, in main code

class PyramidLevelL1:
    depth_map: np.ndarray
    vertex_map: np.ndarray
    normal_map: np.ndarray

    def __init__(self) -> None: ...
    
class PyramidLevelL2:
    depth_map: np.ndarray
    vertex_map: np.ndarray
    normal_map: np.ndarray

    def __init__(self) -> None: ...

class PyramidLevelL3:
    depth_map: np.ndarray
    vertex_map: np.ndarray
    normal_map: np.ndarray

    def __init__(self) -> None: ...

class ProcessedFrame:
    validity_mask: np.ndarray
    raw_depth: np.ndarray
    l1: PyramidLevelL1
    l2: PyramidLevelL2
    l3: PyramidLevelL3

    def __init__(self) -> None: ...


class Device:
    @staticmethod
    def K() -> np.ndarray:
        """ Returns a 3x3 NumPy array representing K matrix """
        ...

    @staticmethod
    def K2() -> np.ndarray:
        """ Returns a 3x3 NumPy array representing K2 matrix """
        ...

    @staticmethod
    def K3() -> np.ndarray:
        """ Returns a 3x3 NumPy array representing K3 matrix """
        ...

    @staticmethod
    def maxDepth() -> int:
        """ Returns the maximum depth value """
        ...

    @staticmethod
    def minDepth() -> int:
        """ Returns the minimum depth value """
        ...

    @staticmethod
    def set_cxx_running(is_running: bool) -> void:
        """ Ends the C++ Code execution """
        ...
    
    @staticmethod
    def set_python_processing(is_running: bool) -> void:
        """ Sets Python execution status for callback """
        ...


def PopFrame() -> ProcessedFrame: ...
def Init(pythonCallback: Callable[[], None]) -> int: ...
