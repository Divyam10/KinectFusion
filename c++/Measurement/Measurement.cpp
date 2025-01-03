// Measurement.cpp : Defines the entry point for the application.
//

#include "Measurement.h"

static void InitKinectFusion(shared_ptr<KinectFusion> kinectFusion, promise<void>& initPromise);
static void HandleKeyPresses(atomic<bool>& isRunning);

int main()
{
	atomic<bool> isRunning = true; //Set before creation of threads
	thread kinectFusionThread;
	shared_ptr<KinectFusion> kinectFusion;
	promise<void> initPromise;
	future<void> initFuture = initPromise.get_future();

	try
	{
		//Construct in main thread
		kinectFusion = make_shared<KinectFusion>(isRunning);

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

static void InitKinectFusion(shared_ptr<KinectFusion> kinectFusion, promise<void>& initPromise)
{
	kinectFusion->Init(initPromise);
}

//TODO Check and work on this!
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
