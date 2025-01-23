// Measurement.cpp : Defines the entry point for the application.
#pragma once
#include "KinectFusion.h"
#include <OpenNI.h>
#include <thread>
#include <memory>
#include <atomic>

#include <chrono> //TODO remove

using namespace std;
using namespace chrono;

static void InitKinectFusion(shared_ptr<KinectFusion> kinectFusion, promise<void>& initPromise);
static void HandleKeyPresses(atomic<bool>& isRunning);

auto previousTime = high_resolution_clock::now();

/*
* Called by Python with "OnFrame" Callback
*/
static int CxxMain(function<void()> pythonCallback) {

	atomic<bool> isRunning = true; //Set before creation of threads
	thread kinectFusionThread;
	shared_ptr<KinectFusion> kinectFusion;
	promise<void> initPromise;
	future<void> initFuture = initPromise.get_future();

	try
	{
		//Construct in main thread
		kinectFusion = make_shared<KinectFusion>(isRunning, pythonCallback);

		//Init on worker thread
		kinectFusionThread = thread(InitKinectFusion, kinectFusion, ref(initPromise));

		//Check for Exceptions
		initFuture.get();

		HandleKeyPresses(isRunning);

		if (kinectFusionThread.joinable())
			kinectFusionThread.join();
	}
	catch (const std::exception& ex)
	{
		isRunning = false;
		cerr << "Error: " << ex.what() << std::endl;

		if (kinectFusionThread.joinable())
			kinectFusionThread.join();

		return -1;
	}

	return 0;
}


/*
* Debugging functions
*/

/*********************************************/
static void TestPythonCallbackFunc() {
	//const ProcessedFrame& processedFrame = 
	auto currentTime = high_resolution_clock::now();
	cout << "Ms/Frame:  " << duration_cast<milliseconds>(currentTime - previousTime).count() << endl; // ~40ms per call (frame)
	previousTime = currentTime;

}

int main()
{
	return CxxMain([]() { TestPythonCallbackFunc(); });
}
/*********************************************/


/*
* KinectFusion logic initialized by worker thread
*/
static void InitKinectFusion(shared_ptr<KinectFusion> kinectFusion, promise<void>& initPromise)
{
	kinectFusion->Init(initPromise);
}

//TODO Check and work on this!
/*
* Key Input handled by worker thread
*/

static void HandleKeyPresses(atomic<bool>& isRunning)
{
	//TODO "Please Press Q to exit"

	while (isRunning)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(100));

		char key = cin.get();

		if (key == 'q' || key == 'Q')
		{
			cout << "Shutdown initiated..." << endl;
			isRunning = false;
		}
	}
	cout << "INPUT HANDLING OVER" << endl;
}
