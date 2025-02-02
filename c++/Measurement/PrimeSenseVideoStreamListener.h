#pragma once
#include <OpenNI.h>
#include <functional>
#include "Structs.h"
#include <memory>
#include <queue>

using namespace openni;
using namespace std;

class PrimeSenseVideoStreamListener : public VideoStream::NewFrameListener
{
public:
	PrimeSenseVideoStreamListener(function<void()> callback, shared_ptr<queue<unique_ptr<VideoFrameRef>>> frameQueue) : callback(callback), frameQueue(frameQueue) {
		
	}

	function<void()> callback;
	shared_ptr<queue<unique_ptr<VideoFrameRef>>> frameQueue;

	void onNewFrame(VideoStream& videoStream) override
	{
		unique_ptr<VideoFrameRef> videoFrame = make_unique<VideoFrameRef>();
		videoStream.readFrame(videoFrame.get());

		frameQueue->push(move(videoFrame));
		callback();
	}
};
