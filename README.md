# KinectFusion
## Measurement
### Installation
- To create the Python module with wrapped C++ code: 
	- Use the Project in the c++ folder
	- Install all Libs (except for CUDA) in the Libs folder (or change CmakeList.txt)
	- Install OpenNI2 and drivers from [OpenNI2](https://structure.io/openni/?srsltid=AfmBOopxqKcIPaKikYaKtega0IkYstv0SPbppemkyJ2OQprldQCDl6Ha)
	- Install CUDA from [CUDA](https://developer.nvidia.com/cuda-downloads)
	- Install Pybind11 from [Pybind11](https://github.com/pybind/pybind11)
	- Modify CmakeLists.txt to own the libs
	- Build
	- Output is a shared library .pyd, .so, .dylib (depending on system)
	- A Typedef file .pyi is provided in the main folder and needs to be next to the shared lib file in order to provide type hints.
	- For windows: Copy OpenNI2 and CUDA .dlls to same folder as .pyd file
	- For windows?: Copy OpenNI2 Folder containing Drivers (.dlls) to same folder as .pyd file
	- Import shared library using "import MeasurementModule" into python program.
  