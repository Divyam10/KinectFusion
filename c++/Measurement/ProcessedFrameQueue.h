#pragma once

#include <memory>
#include <atomic>
#include <functional>
#include <queue>
#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>

namespace py = pybind11;

class ProcessedFrameQueue {
public:
    atomic<bool> isPythonProcessing{ false };
    // Create Singleton
    static std::shared_ptr<ProcessedFrameQueue> Init(
        std::atomic<bool>& isRunning,
        unsigned int queueSize) {

        std::call_once(initFlag, [&]() {
            instance = std::make_shared<ProcessedFrameQueue>(isRunning, queueSize);
            });
        isInitialized.store(true, std::memory_order_release);
        return instance;  
    }

    // Access Singleton
    static std::shared_ptr<ProcessedFrameQueue> Instance() {
        if (!instance) {
            py::gil_scoped_acquire acquire;
            PySys_WriteStdout("C++ Queue Instance is Null!\n");
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
        this->callbackFunc = pythonCallback;
    }

    void Push(const ProcessedFrame& item) {
        std::unique_lock<std::mutex> l(m);

        while (queue.size() >= queueSize) {
            queue.pop();
        }

        queue.push(item);

        l.unlock();

        if (!this->isPythonProcessing) {
            py::gil_scoped_acquire acquire;
            PySys_WriteStdout("C++ Callback called\n");
            this->callbackFunc();
        }
    }

    optional<ProcessedFrame> Pop() {
        std::unique_lock<std::mutex> l(m);

        if (queue.empty())
            return nullopt;

        ProcessedFrame item = queue.front();
        queue.pop();

        isPythonProcessing = true;

        l.unlock();

        py::gil_scoped_acquire acquire;
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
static optional<ProcessedFrame> PopFromQueue() {
    auto queue = ProcessedFrameQueue::Instance();
    return queue->Pop();
}

// Python accessible method to set callback on 2 frames
static bool TrySetFrameCallback(std::function<void()> pythonCallback) {
    py::gil_scoped_acquire acquire;
    PySys_WriteStdout("C++ Checking initialization\n");

    if (!ProcessedFrameQueue::IsInitialized()) {
        return false;
    }

    PySys_WriteStdout("C++ Now Initialized!\n");

    auto queue = ProcessedFrameQueue::Instance();

    if (queue == nullptr || queue == NULL)
        PySys_WriteStdout("C++ Queue NULL!\n");
    
    queue->SetCallback(pythonCallback);

    PySys_WriteStdout("C++ Initialized and callback set\n");
    return true;
}

// TODO I think this should always set it to false when its done...
static void SetPythonProcessing(bool isProcessing) {
    auto queue = ProcessedFrameQueue::Instance();
    queue->isPythonProcessing = isProcessing;
}
