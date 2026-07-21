#ifndef OPS_MLIR_RUNTIME_PROFILER_H

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ops_mlir {

struct KernelTiming {
    std::string kernel_name;
    double milliseconds;
};

class KernelProfiler {
public:
    static KernelProfiler &instance();

    class ScopedTimer {
    public:
        ScopedTimer(KernelProfiler &profiler, std::string kernelName) : profiler_(profiler), name_(std::move(kernelName)), start_(std::chrono::steady_clock::now()) {}

        ~ScopedTimer() {
            auto end = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start_).count();
            profiler_.record(name_, ms);
        }
    private:
        KernelProfiler &profiler_;
        std::string name_;
        std::chrono::steady_clock::time_point start_;
    };

    void record(const std::string &kernelName, double ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.push_back({kernelName, ms});
    }

    const std::vector<KernelTiming> &records() const { return records_; }

    void report() const;
    void writeCsv(const std::string &path) const;

private:
    std::mutex mutex_;
    std::vector<KernelTiming> records_;
};


} // namespace ops_mlir




#endif // OPS_MLIR_RUNTIME_CORE_H