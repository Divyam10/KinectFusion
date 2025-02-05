from typing import Callable
import numpy as np
# Numpy needs to be imported like this: import numpy as np, in main code

class ProcessedFrame:
    color_map: np.ndarray
    raw_depth: np.ndarray
    depth_map_l1: np.ndarray
    depth_map_l2: np.ndarray
    depth_map_l3: np.ndarray

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
    def set_cxx_running(is_running: bool):
        """ Ends the C++ Code execution """


def PopFrame() -> ProcessedFrame: ...
def Init() -> int: ...
def FrameCallback(pythonCallback: Callable[[], None]) -> bool: ...
def StartProcessingThread(): ...
