#ifndef OPS_MLIR_RUNTIME_PROFILER_H
#define OPS_MLIR_RUNTIME_PROFILER_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

namespace ops_mlir {

struct KernelTiming {
    std::string kernel_name;
    double milliseconds;
};

struct KernelStats {
    std::string kernel_name;
    int count = 0;
    double total_ms = 0.0;
    double avg_ms() const { return count ? total_ms / count: 0.0; };
};

class KernelProfiler {
public:
    using TimePoint = std::chrono::steady_clock::time_point;

    [[nodiscard]] TimePoint start() const { return std::chrono::steady_clock::now(); }

    void end(const std::string &kernelName, TimePoint startTime) {
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - startTime)
                        .count();
        record(kernelName, ms);
    }

    void record(const std::string &kernelName, double ms) {
        records_.push_back({kernelName, ms});
    }

    const std::vector<KernelTiming> &records() const { return records_; }

    std::vector<KernelStats> aggregate() const;

    void report() const;
    void writeCsv(const std::string &path) const;

private:
    std::vector<KernelTiming> records_;
};

} // namespace ops_mlir

#endif // OPS_MLIR_RUNTIME_PROFILER_H
