template <typename T>
class QueueWrapper {
public:
    QueueWrapper(
        mutex& m,
        condition_variable& cv,
        atomic<bool>& isRunning,
        function<void(const T&)> callback)
        : m(m), cv(cv), isRunning(isRunning), callbackFunc(callback) {}

    ~QueueWrapper() {}

    void Push(T item)
    {
        unique_lock<mutex> l(m);
        this->queue.push(std::move(item));
        cv.notify_one();
    }

    void WaitForPopAndCallback()
    {
        unique_lock<mutex> l(m);
        while (queue.empty() && isRunning.load())
        {
            cv.wait(l, [this] { return !queue.empty() || !isRunning.load(); });
        }

        if (isRunning.load()) {
            auto item = this->queue.front();
            this->queue.pop();
            callbackFunc(item);
        }
    }

private:
    queue<T> queue;
    mutex& m;
    condition_variable& cv;
    atomic<bool>& isRunning;
    function<void(const T&)> callbackFunc; 
};
