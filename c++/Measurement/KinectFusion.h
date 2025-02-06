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
	static shared_ptr<KinectFusion> KinectFusion::instance;
	unique_ptr<Device> device;
	unique_ptr<VideoStream> videoColorStream;
	unique_ptr<VideoStream> videoDepthStream;
	shared_ptr <PrimeSenseVideoStreamListener> colorFrameListener;
	shared_ptr <PrimeSenseVideoStreamListener> depthFrameListener;
	shared_ptr<queue<unique_ptr<VideoFrameRef>>> colorFrameQueue;
	shared_ptr<queue<unique_ptr<VideoFrameRef>>> depthFrameQueue;
	unique_ptr<queue<unique_ptr<FrameTuple>>> frameTupleQueue;
	shared_ptr<ProcessedFrameQueue> processedFramesQueue; //Passed to Python Shared -> NEEDS to be copyable for python / pybind
	mutex frame_queue_mtx;
	condition_variable& frame_cv;
	VideoMode depthVideoMode; 
	VideoMode colorVideoMode;
	array<float, 9> K; //Passed to Python
	array<float, 9> K2; //Passed to Python
	array<float, 9> K3; //Passed to Python
	array<float, 9> K_inv;
	const unsigned int queueSizeLimit = 30; //Adjustable!
	const unsigned int sigmaSpatial = 10;  //Adjustable!
	const int sigmaRange = 100;  //Adjustable!
	atomic<bool>& isRunning;

	KinectFusion(atomic<bool>& isRunning, std::condition_variable& frame_cv)
		: colorFrameQueue(make_shared<queue<unique_ptr<VideoFrameRef>>>()),
		depthFrameQueue(make_shared<queue<unique_ptr<VideoFrameRef>>>()),
		frameTupleQueue(make_unique<queue<unique_ptr<FrameTuple>>>()),
		isRunning(isRunning),
		//Init Queue Singleton
		processedFramesQueue(ProcessedFrameQueue::Init(isRunning, queueSizeLimit)),
		frame_cv(frame_cv)
	{
		//TODO REALLY REALLY UGLY NO SAFEGUARDS
		if (instance != nullptr)
		{
			throw runtime_error("Based Singleton Error");
			return;
		}

		instance = shared_ptr<KinectFusion>(this);
	};


	~KinectFusion() 
	{
		isRunning = false;

		if (videoColorStream) 
			videoColorStream->destroy();

		if (videoDepthStream)
			videoDepthStream->destroy();

		if (device) 
			device->close();

		OpenNI::shutdown();
	}


	//Call this from another thread
	int Init(promise<void>& initPromise)
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
		py::gil_scoped_acquire acquire;
		PySys_WriteStdout("Device Init Succesfull!\n");
		return 0;
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

		this->depthVideoMode = videoDepthStream->getVideoMode();
		this->depthVideoMode.setResolution(cst::width, cst::height);

		if (videoDepthStream->setVideoMode(this->depthVideoMode) != openni::STATUS_OK) {
			std::cerr << "Failed to set depth video mode: " << openni::OpenNI::getExtendedError() << std::endl;
		}

		this->colorVideoMode = videoColorStream->getVideoMode();
		this->colorVideoMode.setResolution(cst::width, cst::height);

		if (videoColorStream->setVideoMode(this->colorVideoMode) != openni::STATUS_OK) {
			std::cerr << "Failed to set color video mode: " << openni::OpenNI::getExtendedError() << std::endl;
		}

		const float focalX = float(depthVideoMode.getResolutionX()) / 2.f * tanf(videoDepthStream->getHorizontalFieldOfView() / 2.f);
		const float focalY = float(depthVideoMode.getResolutionY()) / 2.f * tanf(videoDepthStream->getVerticalFieldOfView() / 2.f);
		const float princPointX = float(cst::width / 2);
		const float princPointY = float(cst::height / 2);

		//Create inverse of intrinsic K as 1D array
		Eigen::Matrix3f K;
		K << focalX, 0.f, princPointX,
			0.0, focalY, princPointY,
			0.0, 0.0, 1.0;

		Eigen::Matrix3f K2;
		K2 << 0.5f * focalX, 0.f, princPointX * 0.5f,
			0.0, 0.5f * focalY, princPointY * 0.5f,
			0.0, 0.0, 1.0;

		Eigen::Matrix3f K3;
		K3 << 0.25f * focalX, 0.f, princPointX * 0.25f,
			0.0, 0.25f * focalY, princPointY * 0.25f,
			0.0, 0.0, 1.0;

		auto KInv = K.inverse().eval();

		for (int i = 0; i < 9; ++i) {
			this->K_inv[i] = KInv(i / 3, i % 3);

			// Pass K to Python
			this->K[i] = K(i / 3, i % 3);
			this->K2[i] = K2(i / 3, i % 3);
			this->K3[i] = K3(i / 3, i % 3);
		}

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
		if (!this->isRunning)
			return;

		//Critical Section
		{
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
				cout << "FrameDiff != 0!" << endl;
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

			//Notify the processing Thread(s)
			frame_cv.notify_all();
		}
	}


	void ProcessFrames(std::condition_variable& frame_cv)
	{
		{
			py::gil_scoped_acquire acquire;
			PySys_WriteStdout("c++ started processing\n");
		}
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



	/*
	*   RGB888 = 3 bytes (rgb) per pixel
	*   Each index is R->G->B->R...
	*   PIXEL_FORMAT_DEPTH_1_MM(?) -> Resolution of Data in mm
	*   2 Bytes to store max common sensor distances, each index is depth value
	*/
	void ProcessFrame(unique_ptr<FrameTuple> frame)
	{
		//Copy original data
		shared_ptr<array<uint16_t, cst::pixelCount>> depthImage = make_shared<array<uint16_t, cst::pixelCount>>();
		memcpy(depthImage->data(), frame->depthFrame->getData(), cst::pixelCount * sizeof(uint16_t));

		shared_ptr<array<uint8_t, cst::pixelCountXYZ>> colorImage = make_shared<array<uint8_t, cst::pixelCountXYZ>>();
		memcpy(colorImage->data(), frame->colorFrame->getData(), cst::pixelCountXYZ * sizeof(uint8_t));

		shared_ptr<array<uint16_t, cst::pixelCount>> depthImageFiltered = make_shared<array<uint16_t, cst::pixelCount>>();

		shared_ptr<array<bool, cst::pixelCount>> vertexValidityMask = make_shared<array<bool, cst::pixelCount>>();

		//Init all vertices as valid
		vertexValidityMask->fill(true); 

		//TODO adjust kernels to ignore invalid depths using vertexValidityMap
		LaunchBilateralFilteringKernel(
			depthImage->data(),
			depthImageFiltered->data(),
			vertexValidityMask->data(),
			cst::pixelCount,
			cst::width,
			cst::height,
			cst::minDepth,
			cst::maxDepth,
			this->sigmaSpatial,
			this->sigmaRange);

		//TODO move this inside functions probably
		const unsigned int blockSize = 2;

		shared_ptr<array<uint16_t, cst::pixelCountL2>> depthImageFilteredL2 = make_shared<array<uint16_t, cst::pixelCountL2>>();
		shared_ptr<array<uint16_t, cst::pixelCountL3>> depthImageFilteredL3 = make_shared<array<uint16_t, cst::pixelCountL3>>();

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

		processedFramesQueue->Push(
			ProcessedFrame
			{
				move(colorImage),
				move(depthImage),
				move(depthImageFiltered),
				move(depthImageFilteredL2),
				move(depthImageFilteredL3),
			}
		);
	}

	//TODO cleanup lol
	static array<float, 9> getK() {
		return instance->K;
	}

	static array<float, 9> getK2() {
		return instance->K2;
	}

	static array<float, 9> getK3() {
		return instance->K3;
	}

	static void SetCXXRunning(bool isRunning) {
		instance->isRunning = isRunning;
	}
};


//Static init
shared_ptr<KinectFusion> KinectFusion::instance = nullptr;
