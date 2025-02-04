#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/numpy.h> 
#include <pybind11/functional.h>
#include "Structs.h"
#include "ProcessedFrameQueue.h"
#include "Measurement.h"
#include <vector>  
#include <cstddef>  

namespace py = pybind11;

/*
* Function that converts Array to Numpy Array on property call.
*/
template <typename T, size_t N>
py::array_t<T> Array2Numpy(const std::array<T, N>& map, size_t height, size_t width) {
    return py::array_t<T>(
        { height, width },  // Shape
        { width * sizeof(T), sizeof(T) },  // Strides
        map.data()  // Data
    );
}

template <typename T, size_t N>
py::array_t<T> Array2NumpyColor(const std::array<T, N>& map, size_t height, size_t width) {
    // Define shape and strides
    std::vector<size_t> shape = { height, width, 3 }; // (height, width, 3 color channels)
    std::vector<size_t> strides = { width * 3 * sizeof(T), 3 * sizeof(T), sizeof(T) }; // Strides for 3 channels (RGB)

    // Create the numpy array with the given shape, strides, and data pointer
    py::array_t<T> arr(shape, strides, map.data());

    return arr;
}



PYBIND11_MODULE(MeasurementModule, m) {
    py::class_<PyramidLevel<cst::pixelCount>>(m, "PyramidLevelL1")
        .def(py::init<>())
        .def_property_readonly("depth_map", [](const PyramidLevel<cst::pixelCount>& self) { return Array2Numpy(*self.depthMap, cst::height, cst::width); })
        .def_property_readonly("vertex_map", [](const PyramidLevel<cst::pixelCountXYZ>& self) { return Array2Numpy(*self.vertexMap, cst::height, cst::widthXYZ); })
        .def_property_readonly("normal_map", [](const PyramidLevel<cst::pixelCountXYZ>& self) { return Array2Numpy(*self.normalMap, cst::height, cst::widthXYZ); });

    py::class_<PyramidLevel<cst::pixelCountL2>>(m, "PyramidLevelL2")
        .def(py::init<>())
        .def_property_readonly("depth_map", [](const PyramidLevel<cst::pixelCountL2>& self) { return Array2Numpy(*self.depthMap, cst::heightL2, cst::widthL2); })
        .def_property_readonly("vertex_map", [](const PyramidLevel<cst::pixelCountXYZL2>& self) { return Array2Numpy(*self.vertexMap, cst::heightL2, cst::widthL2XYZ); })
        .def_property_readonly("normal_map", [](const PyramidLevel<cst::pixelCountXYZL2>& self) { return Array2Numpy(*self.normalMap, cst::heightL2, cst::widthL2XYZ); });

    py::class_<PyramidLevel<cst::pixelCountL3>>(m, "PyramidLevelL3")
        .def(py::init<>())
        .def_property_readonly("depth_map", [](const PyramidLevel<cst::pixelCountL3>& self) { return Array2Numpy(*self.depthMap, cst::heightL3, cst::widthL3); })
        .def_property_readonly("vertex_map", [](const PyramidLevel<cst::pixelCountXYZL3>& self) { return Array2Numpy(*self.vertexMap, cst::heightL3, cst::widthL3XYZ); })
        .def_property_readonly("normal_map", [](const PyramidLevel<cst::pixelCountXYZL3>& self) { return Array2Numpy(*self.normalMap, cst::heightL3, cst::widthL3XYZ); });

    py::class_<ProcessedFrame>(m, "ProcessedFrame")
        .def(py::init<>())
        .def_property_readonly("validity_mask", [](const ProcessedFrame& self) { return Array2Numpy(*self.validityMask, cst::height, cst::width); })
        .def_property_readonly("raw_depth", [](const ProcessedFrame& self) { return Array2Numpy(*self.rawDepth, cst::height, cst::width); })
        .def_property_readonly("color_map", [](const ProcessedFrame& self) { return Array2NumpyColor(*self.colorMap, cst::height, cst::widthXYZ); })
        .def_property_readonly("l1", [](const ProcessedFrame& self) { return py::cast(self.l1); })
        .def_property_readonly("l2", [](const ProcessedFrame& self) { return py::cast(self.l2); })
        .def_property_readonly("l3", [](const ProcessedFrame& self) { return py::cast(self.l3); });

    m.def("PopFrame", &PopFromQueue),
        R"pbdoc(
            Pops a new processed frame from the queue.
    )pbdoc";

    m.def("Init", &CxxMain,
    R"pbdoc(
            Starts the C++ program with a Python callback function.
            The callback is executed from within the C++ program when 2 frames are available in the framequeue.
    )pbdoc");

    m.def("FrameCallback", &TrySetFrameCallback, py::arg("pythonCallback"),
    R"pbdoc(
            This callback is executed once, from within the C++ program, when 2 frames are available in the framequeue.
    )pbdoc");

    m.def("StartProcessingThread", &StartProcessingThread,
        R"pbdoc(
            Starts StartProcessingThread
    )pbdoc");


    py::class_<KinectFusion>(m, "Device")
        .def_static("K", []() { return Array2Numpy(KinectFusion::getK(), 3, 3); })
        .def_static("K2", []() { return Array2Numpy(KinectFusion::getK2(), 3, 3); })
        .def_static("K3", []() { return Array2Numpy(KinectFusion::getK3(), 3, 3); })
        .def_static("maxDepth", &KinectFusion::getMaxDepth)
        .def_static("minDepth", &KinectFusion::getMinDepth)
        .def_static("set_cxx_running", &KinectFusion::SetCXXRunning);
}
