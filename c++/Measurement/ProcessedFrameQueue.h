#pragma once

#include <memory>
#include <atomic>
#include <functional>
#include <queue>
#include <mutex>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class ProcessedFrameQueue {
public:

    // Create Singleton
    static std::shared_ptr<ProcessedFrameQueue> Init(
        std::atomic<bool>& isRunning,
        unsigned int queueSize) {

        std::call_once(initFlag, [&]() {
            instance = std::make_shared<ProcessedFrameQueue>(isRunning, queueSize);
            });
        isInitialized.store(true, std::memory_order_release);
        //PySys_WriteStdout("QUEUE INITIALIZED\n");
        return instance;  
    }

    // Access Singleton
    static std::shared_ptr<ProcessedFrameQueue> Instance() {
        if (!instance) {
            //PySys_WriteStdout("HORRIBLE YIKES! cant access Instance of framesqueue before init\n");
            throw std::runtime_error("MutexQueue not initialized. Call Init() first.\n");
        }
        return instance;
    }

    static bool IsInitialized() {
        return isInitialized.load(std::memory_order_acquire);
    }

    // Constructor TODO: SHould be private...
    ProcessedFrameQueue(std::atomic<bool>& isRunning, unsigned int queueSize)
        : isRunning(isRunning), queueSize(queueSize){
    }

    ~ProcessedFrameQueue() {}

    void SetCallback(function<void()> pythonCallback) {
        //PySys_WriteStdout("Setting Callback...\n");
        if (!pythonCallback) {
            //PySys_WriteStdout("NULLLL py\n");
        }
        /*
        this->callbackFunc = [pythonCallback]() {
            //PySys_WriteStdout("Inside callback function.\n");

            //py::gil_scoped_acquire acquire;

            pythonCallback();

            //PySys_WriteStdout("Callback executed successfully.\n");
            };
        */
        
        this->callbackFunc = pythonCallback;
        
        //PySys_WriteStdout("Callback set\n");
    }

    void Push(const ProcessedFrame& item) {
        {
            py::gil_scoped_acquire acquire;
            PySys_WriteStdout("Push\n");
        }
        //PySys_WriteStdout("PUSH! check\n");
        std::unique_lock<std::mutex> l(m);

        //if (!this->callbackFunc)
            //return;

        //PySys_WriteStdout("PUSH!\n");

        while (queue.size() >= queueSize) {
            //PySys_WriteStdout("POPPING LIKE CRAZY!\n");
            queue.pop();
        }

        queue.push(item);

        //PySys_WriteStdout("Element count: " + this->elementCumCount);
        //PySys_WriteStdout("\n");

        //On 2 Frames call callback once, dont increase variable further
        if (this->elementCumCount < 1) {
            //PySys_WriteStdout("Element count < 1\n");
            this->elementCumCount++;
        }
        else if (this->elementCumCount == 1){
            //PySys_WriteStdout("Element count == 2\n");
            elementCumCount++;

            //if (callbackFunc == nullptr || callbackFunc == NULL) {
                //PySys_WriteStdout("Callback function is NULL!\n");
               // return;
            //}

            l.unlock();
            //PySys_WriteStdout("After unlock and before cb\n");
            py::gil_scoped_acquire acquire;
            PySys_WriteStdout("Callback le epic try.\n");
            this->callbackFunc();
            PySys_WriteStdout("Callback executed successfully.\n");
        }
    }

    ProcessedFrame Pop() {
        //PySys_WriteStdout("Sounds like pop\n");
        std::unique_lock<std::mutex> l(m);

        if (queue.empty())
        {
            //PySys_WriteStdout("THIS IS REALLY TERRIBLE! no frames in queue\n");
            return ProcessedFrame(); // TODO ? Return default object if queue is empty
        }
        //PySys_WriteStdout("Actual pop\n");
        ProcessedFrame item = queue.front();
        queue.pop();

        l.unlock();
        //PySys_WriteStdout("before acp\n");
        //py::gil_scoped_acquire acquire;
        //PySys_WriteStdout("popppppp\n");

        return item;
    }

private:
    // Singleton Instance
    static std::shared_ptr<ProcessedFrameQueue> instance;

    // Thread safe init
    static std::once_flag initFlag;  

    static std::atomic<bool> isInitialized;

    // Singleton safety
    ProcessedFrameQueue(const ProcessedFrameQueue&) = delete;
    ProcessedFrameQueue& operator=(const ProcessedFrameQueue&) = delete;

    std::queue<ProcessedFrame> queue;
    std::mutex m;
    std::atomic<bool>& isRunning;
    std::function<void()> callbackFunc;
    unsigned int queueSize;
    int elementCumCount = 0;
};

// Static instance pointer & once_flag
std::shared_ptr<ProcessedFrameQueue> ProcessedFrameQueue::instance = nullptr;
std::once_flag ProcessedFrameQueue::initFlag;
std::atomic<bool> ProcessedFrameQueue::isInitialized = false;


// Python accessible method to pop a frame from the queue
static ProcessedFrame PopFromQueue() {
    auto queue = ProcessedFrameQueue::Instance();
    //PySys_WriteStdout("Pop maybe\n");
    return queue->Pop();
}

// Python accessible method to set callback on 2 frames
static bool TrySetFrameCallback(std::function<void()> pythonCallback) {
    py::gil_scoped_acquire acquire;
    PySys_WriteStdout("Checking initialization\n");
    if (!ProcessedFrameQueue::IsInitialized()) {
        PySys_WriteStdout("Not yet!\n");
        return false;
    }

    PySys_WriteStdout("Now Initialized!\n");

    auto queue = ProcessedFrameQueue::Instance();

    if (queue == nullptr || queue == NULL)
        PySys_WriteStdout("Queue NULL!\n");
    
    queue->SetCallback(pythonCallback);

    PySys_WriteStdout("Initialized and callback set\n");
    return true;
}
