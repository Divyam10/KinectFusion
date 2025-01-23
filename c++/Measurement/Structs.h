#pragma once
#include <OpenNI.h>
#include <memory>
#include <array>
#include "Constants.h"

using namespace openni;
using namespace std;
namespace cst = ScanMoCap;

//TODO maybe switch solution, to not hardcode, but this is fine for us


struct FrameTuple
{
	unique_ptr<VideoFrameRef> colorFrame;
	unique_ptr<VideoFrameRef> depthFrame;
};

//Used shared_ptr for pybind11...
template <size_t pixelCount>
struct PyramidLevel {
	shared_ptr<std::array<uint16_t, pixelCount>> depthMap;
	shared_ptr<std::array<uint16_t, pixelCount * 3>> vertexMap;
	shared_ptr<std::array<uint16_t, pixelCount * 3>> normalMap;
};

struct ProcessedFrame
{
	shared_ptr<std::array<bool, cst::pixelCount>> validityMask; //Used shared_ptr for pybind11...
	PyramidLevel<cst::pixelCount> l1;
	PyramidLevel<cst::pixelCountL2> l2;
	PyramidLevel<cst::pixelCountL3> l3;
};
