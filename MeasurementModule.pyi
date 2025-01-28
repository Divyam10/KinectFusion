from typing import Callable
import numpy

class PyramidLevelL1:
    depth_map: "numpy.ndarray"
    vertex_map: "numpy.ndarray"
    normal_map: "numpy.ndarray"

    def __init__(self) -> None: ...
    
class PyramidLevelL2:
    depth_map: "numpy.ndarray"
    vertex_map: "numpy.ndarray"
    normal_map: "numpy.ndarray"

    def __init__(self) -> None: ...

class PyramidLevelL3:
    depth_map: "numpy.ndarray"
    vertex_map: "numpy.ndarray"
    normal_map: "numpy.ndarray"

    def __init__(self) -> None: ...

class ProcessedFrame:
    validity_mask: "numpy.ndarray"
    l1: PyramidLevelL1
    l2: PyramidLevelL2
    l3: PyramidLevelL3

    def __init__(self) -> None: ...

def PopFrame() -> ProcessedFrame: ...
def Init(pythonCallback: Callable[[], None]) -> int: ...

class Device:
    K: "numpy.ndarray"
