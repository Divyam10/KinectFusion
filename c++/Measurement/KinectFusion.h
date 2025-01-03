#pragma once
#include <iostream>
#include <OpenNI.h>
#include "PrimeSenseListener.h"
#include <mutex>
#include <condition_variable>
#include <thread>
#include <shared_mutex>
#include <memory>
#include <atomic>
#include <future>
#include "ImageData.h"

using namespace std;
using namespace openni;

class KinectFusion
{
public:
	unique_ptr<Device> device;
	unique_ptr<VideoStream> videoColorStream;
	unique_ptr<VideoStream> videoDepthStream;
	shared_ptr <PrimeSenseVideoStreamListener> colorFrameListener;
	shared_ptr <PrimeSenseVideoStreamListener> depthFrameListener;
	shared_ptr<queue<unique_ptr<VideoFrameRef>>> colorFrameQueue;
	shared_ptr<queue<unique_ptr<VideoFrameRef>>> depthFrameQueue;
	unique_ptr<queue<unique_ptr<FrameTuple>>> frameTupleQueue;
	PixelFormat pixelFormatColor;
	PixelFormat pixelFormatDepth;
	const unsigned int queueSizeLimit = 30; //Adjustable!
	mutex frame_queue_mtx;
	thread gatheringThread;
	thread processingThread;
	condition_variable cv;
	atomic<bool>& isRunning;

	KinectFusion(atomic<bool>& isRunning)
		: colorFrameQueue(make_shared<queue<unique_ptr<VideoFrameRef>>>()),
		depthFrameQueue(make_shared<queue<unique_ptr<VideoFrameRef>>>()),
		frameTupleQueue(make_unique<queue<unique_ptr<FrameTuple>>>()),
		isRunning(isRunning) {}

	~KinectFusion() 
	{
		isRunning = false;

		// Join threads before destruction
		if (this->processingThread.joinable())
			this->processingThread.join(); 

		if (videoColorStream) 
			videoColorStream->destroy();

		if (videoDepthStream)
			videoDepthStream->destroy();

		if (device) 
			device->close();

		OpenNI::shutdown();
	}

	//Call this from another thread
	void Init(promise<void>& initPromise)
	{
		try
		{
			if (OpenNI::initialize() != STATUS_OK)
				throw runtime_error("Failed to initialize OpenNI: " + string(OpenNI::getExtendedError()));

			//Multiple Devices
			/*
			PrimeSenseConnectedListener primeSenseConnectedListener = PrimeSenseConnectedListener([this](unique_ptr<Device> device)
				{
					this->device = move(device);
					OnDeviceFound(*this->device);
				});

			OpenNI::addDeviceConnectedListener(&primeSenseConnectedListener);
			*/


			this->device = make_unique<Device>();

			if (device->open(ANY_DEVICE) != STATUS_OK)
				throw runtime_error("Failed to open device.");

			if (device->setDepthColorSyncEnabled(true) != STATUS_OK)
				throw runtime_error("No DepthColorSync Possible...");

			OnDeviceFound(*this->device);

			initPromise.set_value();
		}
		catch (const exception&)
		{
			initPromise.set_exception(current_exception());
		}
	}

	/**
	* This function gets called once the Device is found.
	* It sets up the handling of the frames and starts the threads.
	**/
	void OnDeviceFound(Device& device)
	{
		this->videoColorStream = make_unique<VideoStream>();

		if(this->videoColorStream->create(device, SENSOR_COLOR) != STATUS_OK)
			throw runtime_error("Failed to create color stream.");

		this->videoDepthStream = make_unique<VideoStream>();

		if (this->videoDepthStream->create(device, SENSOR_DEPTH) != STATUS_OK)
			throw runtime_error("Failed to create depth stream.");

		this->colorFrameListener = make_shared<PrimeSenseVideoStreamListener>([this]() { OnUpdatedQueue(); }, this->colorFrameQueue);

		if(this->videoColorStream->addNewFrameListener(colorFrameListener.get()) != STATUS_OK)
			throw runtime_error("Failed to add color frame listener.");
		
		this->depthFrameListener = make_shared<PrimeSenseVideoStreamListener>([this]() { OnUpdatedQueue(); }, this->depthFrameQueue);

		if(this->videoDepthStream->addNewFrameListener(depthFrameListener.get()) != STATUS_OK)
			throw runtime_error("Failed to add depth frame listener.");


		//Start Frame Processing Thread
		this->processingThread = thread(&KinectFusion::ProcessFrames, this);

		if (this->videoColorStream->start() != STATUS_OK)
			throw runtime_error("Failed to start color stream.");

		if (this->videoDepthStream->start() != STATUS_OK)
			throw runtime_error("Failed to start depth stream.");

		if (device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR) != STATUS_OK)
			throw runtime_error("No ImageRegistrationMode Possible..." + string(OpenNI::getExtendedError()));
	}

	/**
	* This function is called once a depth or color frame is received from the device and added to the respective queue.
	* It adds color & depth frames to a tuple which is saved in the tuple queue for later processing.
	**/
	void OnUpdatedQueue()
	{
		//Critical Section
		{
			if (!this->isRunning)
				return;

			unique_lock<mutex> lock(frame_queue_mtx);
			
			//Wait for 2+ frames
			if (this->colorFrameQueue->empty() || this->depthFrameQueue->empty())
				return;

			unique_ptr <VideoFrameRef> colorFrame = move(this->colorFrameQueue->front());
			colorFrameQueue->pop();

			unique_ptr <VideoFrameRef> depthFrame = move(this->depthFrameQueue->front());
			depthFrameQueue->pop();

			//Tells us about the last frame that came in, at the time that we first got both 1 depth & 1 color frame
			int frameDiff = colorFrame->getFrameIndex() - depthFrame->getFrameIndex();

			// Color queue empty

			while (frameDiff > 0)
			{
				depthFrame = move(this->depthFrameQueue->front());
				depthFrameQueue->pop();
				frameDiff--;
			}
			
			while (frameDiff < 0)
			{
				colorFrame = move(this->colorFrameQueue->front());
				colorFrameQueue->pop();
				frameDiff++;
			}

			//TODO probably removable
			if (frameDiff != 0)
			{
				cout << "WTF Bruh" << endl;
				return;
			}

			//TODO REMOVE
			cout << colorFrame->getFrameIndex() << endl;

			unique_ptr<FrameTuple> currentFrameTuple = make_unique<FrameTuple>();
			currentFrameTuple->colorFrame = move(colorFrame);
			currentFrameTuple->depthFrame = move(depthFrame);

			this->frameTupleQueue->push(move(currentFrameTuple));

			while (this->frameTupleQueue->size() > this->queueSizeLimit)
			{
				this->frameTupleQueue->pop();
			}

			//Notify 1 (of the) processing Thread(s)
			cv.notify_one();
		}
		
	}

	void ProcessFrames()
	{
		unique_ptr<FrameTuple> currentFrame;

		while (isRunning)
		{
			//Critical Section
			{
				//Not locking until there is available new frame and last frame finished. Wait call blocks this thread until signaled.
				unique_lock<mutex> lk(frame_queue_mtx);

				//Avoid spurious wakeups
				while (frameTupleQueue->empty() && isRunning)
				{
					cv.wait(lk, [this] { return !frameTupleQueue->empty() || !isRunning; });
				}

				currentFrame = move(frameTupleQueue->front());
				frameTupleQueue->pop();
			}
			
			//TODO Processing...
			this->ProcessFrame(move(currentFrame));
		}
	}

	void ProcessFrame(unique_ptr<FrameTuple> frame)
	{
		//Bilateral Filtering
		/*
		auto height = frame->colorFrame->getHeight();
		auto width = frame->colorFrame->getWidth();
		cout << "Color H/W: " << height << " " << width << endl;

		auto height2 = frame->colorFrame->getHeight();
		auto width2 = frame->colorFrame->getWidth();
		cout << "Depth H/W: " << height2 << " " << width2 << endl;

		//auto norm = 1 / 3; //3 should be norm constant...
		//auto N_sig = exp(-t ^ 2 * 1 / (sig ^ 2));
		//auto dk_u = 1/
		*/
		
		//TODO remove "Simulates bilateral filtering
		this_thread::sleep_for(chrono::milliseconds(50));



		//TODO maybe add queue + mutex...



		//Backtransformation
		//TODO move height/width since const
		const auto pixelCount = 320 * 240;

		auto colorDataSize = frame->colorFrame->getDataSize();
		auto depthDataSize = frame->depthFrame->getDataSize();

		//RGB888 = 320 × 240 resolution with 3 bytes (rgb) per pixel
		//Each index is R->G->B->R...
		const uint8_t* colorData = reinterpret_cast<const uint8_t*>(frame->colorFrame->getData());

		//TODO check if depth should be limited by provided "max depth"
		// PIXEL_FORMAT_DEPTH_1_MM(?) -> Resolution of Data in mm
		//2 Bytes to store max common sensor distances, each index is depth value 
		const uint16_t* depthData = reinterpret_cast<const uint16_t*>(frame->depthFrame->getData());

		//unique_ptr<ImageData>imageData = make_unique<ImageData>(colorData, depthData);
	}

};
