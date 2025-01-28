from typing import Callable
# Numpy needs to be imported like this: import numpy as np, in main code

class PyramidLevelL1:
    depth_map: "np.ndarray"
    vertex_map: "np.ndarray"
    normal_map: "np.ndarray"

    def __init__(self) -> None: ...
    
class PyramidLevelL2:
    depth_map: "np.ndarray"
    vertex_map: "np.ndarray"
    normal_map: "np.ndarray"

    def __init__(self) -> None: ...

class PyramidLevelL3:
    depth_map: "np.ndarray"
    vertex_map: "np.ndarray"
    normal_map: "np.ndarray"

    def __init__(self) -> None: ...

class ProcessedFrame:
    validity_mask: "np.ndarray"
    l1: PyramidLevelL1
    l2: PyramidLevelL2
    l3: PyramidLevelL3

    def __init__(self) -> None: ...


class Device:
    K: "np.ndarray"
    min_depth: int
    max_depth: int


def PopFrame() -> ProcessedFrame: ...
def Init(pythonCallback: Callable[[], None]) -> int: ...
