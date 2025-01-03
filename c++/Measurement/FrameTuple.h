#pragma once
#include <OpenNI.h>
#include <memory>

using namespace openni;
using namespace std;

struct FrameTuple
{
	unique_ptr<VideoFrameRef> colorFrame;
	unique_ptr<VideoFrameRef> depthFrame;
};
