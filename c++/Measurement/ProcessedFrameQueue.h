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
        std::function<void()> callback,
        unsigned int queueSize,
        atomic<bool>& isPythonProcessing) {

        std::call_once(initFlag, [&]() {
            instance = std::make_shared<ProcessedFrameQueue>(isRunning, callback, queueSize, isPythonProcessing);
            });

        PySys_WriteStdout("QUEUE INITIALIZED\n");
        return instance;  
    }

    // Access Singleton
    static std::shared_ptr<ProcessedFrameQueue> Instance() {
        if (!instance) {
            PySys_WriteStdout("HORRIBLE YIKES! cant access Instance of framesqueue before init\n");
            throw std::runtime_error("MutexQueue not initialized. Call Init() first.");
        }

        return instance;
    }

    // Constructor TODO: SHould be private...
    ProcessedFrameQueue(std::atomic<bool>& isRunning, std::function<void()> callback, unsigned int queueSize, atomic<bool>& isPythonProcessing)
        : isRunning(isRunning), callbackFunc(callback), queueSize(queueSize), isPythonProcessing(isPythonProcessing){
    }

    ~ProcessedFrameQueue() {}


    void Push(const ProcessedFrame& item) {
        std::unique_lock<std::mutex> l(m);

        while (queue.size() >= queueSize) {
            queue.pop();
        }

        queue.push(item);

        l.unlock();

        if (!this->isPythonProcessing) {
            py::gil_scoped_acquire acquire;
            PySys_WriteStdout("Calling back\n");
            callbackFunc();
        }
    }

    ProcessedFrame Pop() {
        std::unique_lock<std::mutex> l(m);

        if (queue.empty())
        {
            PySys_WriteStdout("THIS IS REALLY TERRIBLE! no frames in queue\n");
            return ProcessedFrame(); // TODO ? Return default object if queue is empty
        }

        isPythonProcessing = true;
            
        ProcessedFrame item = queue.front();
        queue.pop();

        l.unlock();

        py::gil_scoped_acquire acquire;

        return item;
    }

private:
    // Singleton Instance
    static std::shared_ptr<ProcessedFrameQueue> instance;

    // Thread safe init
    static std::once_flag initFlag;  

    // Singleton safety
    ProcessedFrameQueue(const ProcessedFrameQueue&) = delete;
    ProcessedFrameQueue& operator=(const ProcessedFrameQueue&) = delete;

    std::queue<ProcessedFrame> queue;
    std::mutex m;
    std::atomic<bool>& isRunning;
    std::function<void()> callbackFunc;
    unsigned int queueSize;
    atomic<bool>& isPythonProcessing;
};

// Static instance pointer & once_flag
std::shared_ptr<ProcessedFrameQueue> ProcessedFrameQueue::instance = nullptr;
std::once_flag ProcessedFrameQueue::initFlag;

// Python accessible method to pop a frame from the queue
static ProcessedFrame PopFromQueue() {
    auto queue = ProcessedFrameQueue::Instance();
    return queue->Pop();
}
