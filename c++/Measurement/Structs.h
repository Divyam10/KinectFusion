#pragma once
#include <OpenNI.h>
#include <memory>
#include <array>
#include "Constants.h"

using namespace openni;
using namespace std;
namespace cst = ScanMoCap;

struct ImageData
{
	const uint8_t* colorData;
	const uint16_t* depthData;

	ImageData(const uint8_t* color, const uint16_t* depth)
		: colorData(color), depthData(depth) {
	}
};

struct FrameTuple
{
	unique_ptr<VideoFrameRef> colorFrame;
	unique_ptr<VideoFrameRef> depthFrame;
};

struct ProcessedFrame
{
	shared_ptr<std::array<uint8_t, cst::pixelCountXYZ>> colorMap;
	shared_ptr<std::array<uint16_t, cst::pixelCount>> rawDepth;
	shared_ptr<std::array<uint16_t, cst::pixelCount>> depthMap_l1;
	shared_ptr<std::array<uint16_t, cst::pixelCountL2>> depthMap_l2;
	shared_ptr<std::array<uint16_t, cst::pixelCountL3>> depthMap_l3;
};
