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
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <Eigen/Dense>
#include "ProcessedFrameQueue.h"
#include "Constants.h"
#include "Structs.h"

namespace cst = ScanMoCap;

//Definitions for C++ Compiler
extern "C" void LaunchBilateralFilteringKernel(
	const uint16_t* depthImage,
	uint16_t* depthImageFiltered,
	bool* vertexValidityMask,
	const unsigned int pixelCount,
	const unsigned int width,
	const unsigned int height,
	const unsigned int minDepth,
	const unsigned int maxDepth,
	const unsigned int sigmaSpatial,
	const unsigned int sigmaRange);

extern "C" void LaunchBackProjectionKernel(
	float* K_inv,
	const uint16_t* depthImageFiltered,
	uint16_t* backProjectedDepthImage,
	const unsigned int pixelCount,
	const unsigned int width,
	const unsigned int height,
	const unsigned int stride);

extern "C" void LaunchBlockAveragingAndSubsampleKernel(
	const uint16_t* depthImageL,
	uint16_t* depthImageLAveraged,
	const unsigned int pixelCountL,
	const unsigned int pixelCountLAveraged,
	const unsigned int widthL,
	const unsigned int heightL,
	const unsigned int blockSize,
	const unsigned int sigmaRange);

extern "C" void LaunchNormalMapKernel(
	const uint16_t* vertexMap,
	uint16_t* normalMap,
	const unsigned int pixelCount,
	const unsigned int width,
	const unsigned int height,
	const unsigned int stride);


class KinectFusion {
public:
	unique_ptr<Device> device;
	unique_ptr<VideoStream> videoColorStream;
	unique_ptr<VideoStream> videoDepthStream;
	shared_ptr <PrimeSenseVideoStreamListener> colorFrameListener;
	shared_ptr <PrimeSenseVideoStreamListener> depthFrameListener;
	shared_ptr<queue<unique_ptr<VideoFrameRef>>> colorFrameQueue;
	shared_ptr<queue<unique_ptr<VideoFrameRef>>> depthFrameQueue;
	unique_ptr<queue<unique_ptr<FrameTuple>>> frameTupleQueue;

	shared_ptr<ProcessedFrameQueue> processedFramesQueue; // Shared -> NEEDS to be copyable for python / pybind

	VideoMode depthVideoMode;
	PixelFormat pixelFormatColor;
	PixelFormat pixelFormatDepth;
	//unsigned int width = 0;
	//unsigned int height = 0;
	//unsigned int pixelCount = 0;
	unsigned int maxDepth = 0;
	unsigned int minDepth = 0;
	array<float, 9> K_inv;
	const unsigned int queueSizeLimit = 30; //Adjustable!
	mutex frame_queue_mtx;
	thread gatheringThread;
	thread processingThread;
	condition_variable frame_cv;
	atomic<bool>& isRunning;
	const unsigned int sigmaSpatial = 5; //TODO might move out of this function, should be the same across program!
	const int sigmaRange = 50; //TODO might move out of this function,should be the same across program!

	KinectFusion(
		atomic<bool>& isRunning,
		function<void()> pythonCallback)
		: colorFrameQueue(make_shared<queue<unique_ptr<VideoFrameRef>>>()),
		depthFrameQueue(make_shared<queue<unique_ptr<VideoFrameRef>>>()),
		frameTupleQueue(make_unique<queue<unique_ptr<FrameTuple>>>()),
		isRunning(isRunning),
		/*
		processedFramesQueue(
			make_shared<MutexQueue<ProcessedFrame>>(
				this->isRunning,
				pythonCallback,
				this->queueSizeLimit) //Might give this separate limit
		){}
		*/
		//Init Queue Singleton
		processedFramesQueue(ProcessedFrameQueue::Init(isRunning, pythonCallback, queueSizeLimit)) {};

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
		PySys_WriteStdout("Device Init Succesfull!\n");
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

		//TODO camera exposure??
		this->depthVideoMode = videoDepthStream->getVideoMode();
		//this->width = this->depthVideoMode.getResolutionX();
		//this->height = this->depthVideoMode.getResolutionY();
		//this->pixelCount = width * height;
		this->maxDepth = videoDepthStream->getMaxPixelValue();
		this->minDepth = videoDepthStream->getMinPixelValue();

		const float focalX = float(depthVideoMode.getResolutionX()) / 2.f * tanf(videoDepthStream->getHorizontalFieldOfView() / 2.f);
		const float focalY = float(depthVideoMode.getResolutionY()) / 2.f * tanf(videoDepthStream->getVerticalFieldOfView() / 2.f);
		const float princPointX = float(cst::width / 2);
		const float princPointY = float(cst::height / 2);

		//Create inverse of intrinsic K as 1D array
		Eigen::Matrix3f tempK;
		tempK << focalX, 0.f, princPointX,
			0.0, focalY, princPointY,
			0.0, 0.0, 1.0;

		tempK = tempK.inverse().eval();

		for (int i = 0; i < 9; ++i) {
			this->K_inv[i] = tempK(i / 3, i % 3); 
		}

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

			while (frameDiff > 0)
			{
				if (this->depthFrameQueue->empty())
					return;

				depthFrame = move(this->depthFrameQueue->front());
				depthFrameQueue->pop();
				frameDiff--;
			}
			
			while (frameDiff < 0)
			{
				if (this->colorFrameQueue->empty())
					return;

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

			unique_ptr<FrameTuple> currentFrameTuple = make_unique<FrameTuple>();
			currentFrameTuple->colorFrame = move(colorFrame);
			currentFrameTuple->depthFrame = move(depthFrame);

			this->frameTupleQueue->push(move(currentFrameTuple));

			while (this->frameTupleQueue->size() > this->queueSizeLimit)
			{
				this->frameTupleQueue->pop();
			}
			//TODO ok. here me out. if this is commented out i cant see any output on the python side?
			PySys_WriteStdout("RawFrame\n");

			//Notify 1 (of the) processing Thread(s)
			frame_cv.notify_one();
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
					frame_cv.wait(lk, [this] { return !frameTupleQueue->empty() || !isRunning; });
				}

				if (frameTupleQueue->empty())
					continue;

				currentFrame = move(frameTupleQueue->front());
				frameTupleQueue->pop();
			}
			
			this->ProcessFrame(move(currentFrame));
		}
	}

	void ProcessFrame(unique_ptr<FrameTuple> frame)
	{
		PySys_WriteStdout("Processing Frame\n");
		//320 × 240 resolution, 76800 pixels in both color & depth
		//RGB888 = 3 bytes (rgb) per pixel
		//Each index is R->G->B->R...
		// PIXEL_FORMAT_DEPTH_1_MM(?) -> Resolution of Data in mm
		//2 Bytes to store max common sensor distances, each index is depth value
		//const uint16_t* depthImage = reinterpret_cast<const uint16_t*>(frame->depthFrame->getData());
		//uint16_t* depthImageFiltered = new uint16_t[pixelCount];
		

		//Copy original data
		shared_ptr<array<uint16_t, cst::pixelCount>> depthImage = make_shared<array<uint16_t, cst::pixelCount>>();
		memcpy(depthImage->data(), frame->depthFrame->getData(), cst::pixelCount * sizeof(uint16_t));

		shared_ptr<array<uint16_t, cst::pixelCount>> depthImageFiltered = make_shared<array<uint16_t, cst::pixelCount>>();

		shared_ptr<array<uint16_t, cst::pixelCountXYZ>> vertexMap = make_shared<array<uint16_t, cst::pixelCountXYZ>>();

		shared_ptr<array<bool, cst::pixelCount>> vertexValidityMask = make_shared<array<bool, cst::pixelCount>>();
		//Init all vertices as valid
		vertexValidityMask->fill(true); 


		//Blocking CUDA function
		//TODO adjust kernels to ignore invalid depths using vertexValidityMap
		LaunchBilateralFilteringKernel(
			depthImage->data(),
			depthImageFiltered->data(),
			vertexValidityMask->data(),
			cst::pixelCount,
			cst::width,
			cst::height,
			this->minDepth,
			this->maxDepth,
			this->sigmaSpatial,
			this->sigmaRange);

		LaunchBackProjectionKernel(
			this->K_inv.data(),
			depthImageFiltered->data(),
			vertexMap->data(),
			cst::pixelCount,
			cst::width,
			cst::height,
			cst::stride);

		//Calculate L=3: DepthImagePyramid, VertexMapPyramid and NormalMapPyramid
		const unsigned int blockSize = 2;

		shared_ptr<array<uint16_t, cst::pixelCountL2>> depthImageFilteredL2 = make_shared<array<uint16_t, cst::pixelCountL2>>();
		shared_ptr<array<uint16_t, cst::pixelCountL3>> depthImageFilteredL3 = make_shared<array<uint16_t, cst::pixelCountL3>>();

		shared_ptr<array<uint16_t, cst::pixelCountXYZL2>> vertexMapL2 = make_shared<array<uint16_t, cst::pixelCountXYZL2>>();
		shared_ptr<array<uint16_t, cst::pixelCountXYZL3>> vertexMapL3 = make_shared<array<uint16_t, cst::pixelCountXYZL3>>();

		shared_ptr<array<uint16_t, cst::pixelCountXYZ>> normalMap = make_shared<array<uint16_t, cst::pixelCountXYZ>>();
		shared_ptr<array<uint16_t, cst::pixelCountXYZL2>> normalMapL2 = make_shared<array<uint16_t, cst::pixelCountXYZL2>>();
		shared_ptr<array<uint16_t, cst::pixelCountXYZL3>> normalMapL3 = make_shared<array<uint16_t, cst::pixelCountXYZL3>>();

		//Blockaveraging and Subsampling to half resolution --> 2x2 blocks
		LaunchBlockAveragingAndSubsampleKernel(
			depthImageFiltered->data(),
			depthImageFilteredL2->data(),
			cst::pixelCount,
			cst::pixelCountL2,
			cst::width,
			cst::height,
			blockSize,
			this->sigmaRange);
		
		LaunchBlockAveragingAndSubsampleKernel(
			depthImageFilteredL2->data(),
			depthImageFilteredL3->data(),
			cst::pixelCountL2,
			cst::pixelCountL3,
			cst::widthL2,
			cst::heightL2,
			blockSize,
			this->sigmaRange);

		LaunchBackProjectionKernel(
			this->K_inv.data(),
			depthImageFilteredL2->data(),
			vertexMapL2->data(),
			cst::pixelCountL2,
			cst::widthL2,
			cst::heightL2,
			cst::stride);

		LaunchBackProjectionKernel(
			this->K_inv.data(),
			depthImageFilteredL3->data(),
			vertexMapL3->data(),
			cst::pixelCountL3,
			cst::widthL3,
			cst::heightL3,
			cst::stride);

		LaunchNormalMapKernel(
			vertexMap->data(),
			normalMap->data(),
			cst::pixelCount,
			cst::width,
			cst::height,
			cst::stride
		);

		LaunchNormalMapKernel(
			vertexMapL2->data(),
			normalMapL2->data(),
			cst::pixelCountL2,
			cst::widthL2,
			cst::heightL2,
			cst::stride
		);

		LaunchNormalMapKernel(
			vertexMapL3->data(),
			normalMapL3->data(),
			cst::pixelCountL3,
			cst::widthL3,
			cst::heightL3,
			cst::stride
		);

		//TODO check if this should be a pointer
		processedFramesQueue->Push(
			ProcessedFrame
			{
				move(vertexValidityMask),
				PyramidLevel<cst::pixelCount>
				{
					move(depthImageFiltered),
					move(vertexMap),
					move(normalMap)
				},
				PyramidLevel<cst::pixelCountL2>
				{
					move(depthImageFilteredL2),
					move(vertexMapL2),
					move(normalMapL2)
				},
				PyramidLevel<cst::pixelCountL3>
				{
					move(depthImageFilteredL3),
					move(vertexMapL3),
					move(normalMapL3)
				}
			}
		);


		//TODO 
		// Queue: make mutex internal (maybe remove cv?
		// Copy queue size limit as done before with mutex to remove/add
		// 
		//Expose Queue.pop to python, not queue.
		//Call this func on python callback
		//
		//MAYBE: Have function in c++ wait for frames (as is hypothetically implemented in queue) then call python cb

		
	}

};
