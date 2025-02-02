#pragma once
#include "KinectFusion.h"
#include <OpenNI.h>
#include <thread>
#include <memory>
#include <atomic>

using namespace std;

static void InitKinectFusion(shared_ptr<KinectFusion> kinectFusion, promise<void>& initPromise);
static void HandleKeyPresses(atomic<bool>& isRunning);


/*
* Called by Python with "OnFrame" Callback
*/
static int CxxMain(function<void()> pythonCallback) {
	atomic<bool> isRunning = true;
	atomic<bool> isPythonProcessing = false;
	thread kinectFusionThread;
	shared_ptr<KinectFusion> kinectFusion;
	promise<void> initPromise;
	future<void> initFuture = initPromise.get_future();

	try
	{
		//Construct in main thread
		kinectFusion = make_shared<KinectFusion>(isRunning, pythonCallback, isPythonProcessing);

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
* KinectFusion logic initialized by worker thread
*/
static void InitKinectFusion(shared_ptr<KinectFusion> kinectFusion, promise<void>& initPromise)
{
	kinectFusion->Init(initPromise);
}


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

/*
//Debugging function
static void TestPythonCallbackFunc() {
	//const ProcessedFrame& processedFrame =
	auto currentTime = high_resolution_clock::now();
	cout << "Ms/Frame:  " << duration_cast<milliseconds>(currentTime - previousTime).count() << endl; // ~40ms per call (frame)
	previousTime = currentTime;
}
*/

int main()
{
	//return CxxMain([]() { TestPythonCallbackFunc(); });
}
