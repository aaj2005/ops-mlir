#include "runtime/KernelProfiler.h"

#include <fstream>
#include <iostream>
#include <unordered_map>

namespace ops_mlir {

    KernelProfiler &KernelProfiler::instance() {
        static KernelProfiler profiler;
        return profiler;
    }

    void KernelProfiler::report() const {
        // Aggregate by kernel name: count, total, average

        struct Stats {
            int count = 0;
            double total_ms = 0.0;
        };

        std::unordered_map<std::string, Stats> byKernel;
        for (const auto &t : records_) {
            auto &s = byKernel[t.kernel_name];
            s.count += 1;
            s.total_ms += t.milliseconds;
        }

        std::cout << "=== Kernel Timing Summary ===\n";
        for (const auto &[name, s] : byKernel) {
            std::cout << name << ": " << s.count << " calls, "
                        << s.total_ms << " ms total, "
                        << (s.total_ms / s.count) << " ms avg\n";
        }
    }

    void KernelProfiler::writeCsv(const std::string &path) const {
        std::ofstream out(path);
        if (!out) {
            std::cerr << "KernelProfiler: failed to open " << path << " for writing\n";
        }

        out << "kernel_name,milliseconds\n";
        for (const auto &t : records_) {
            out << t.kernel_name << "," << t.milliseconds << "\n";
        }
    }
} // namespace ops_mlir