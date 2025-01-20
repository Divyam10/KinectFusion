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

template <size_t pixelCount>
struct PyramidLevel {
	unique_ptr<array<uint16_t, pixelCount>> depthMap;
	unique_ptr<array<uint16_t, pixelCount * 3>> vertexMap;
	unique_ptr<array<uint16_t, pixelCount * 3>> normalMap;
};

struct ProcessedFrame
{
	unique_ptr<array<bool, cst::pixelCount>> validityMask;
	PyramidLevel<cst::pixelCount> l1;
	PyramidLevel<cst::pixelCountL2> l2;
	PyramidLevel<cst::pixelCountL3> l3;
};