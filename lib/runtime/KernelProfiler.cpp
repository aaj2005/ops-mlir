#include "runtime/KernelProfiler.h"

#include <fstream>
#include <iostream>
#include <unordered_map>

namespace ops_mlir {

    KernelProfiler &KernelProfiler::instance() {
        static KernelProfiler profiler;
        return profiler;
    }

    std::vector<KernelStats> KernelProfiler::aggregate() const {
        std::unordered_map<std::string, KernelStats> byKernel;
        for (const auto &t : records_) {
            auto &s = byKernel[t.kernel_name];
            s.kernel_name = t.kernel_name;
            s.count += 1;
            s.total_ms += t.milliseconds;
        }

        std::vector<KernelStats> result;
        result.reserve(byKernel.size());
        for (auto &[name, stats] : byKernel) {
            result.push_back(stats);
        }
        
        return result;
    }

    void KernelProfiler::report() const {
        std::cout << "=== Kernel Timing Summary ===\n";
        for (const auto &s : aggregate()) {
            std::cout << s.kernel_name << ": " << s.count << " calls, "
                        << s.total_ms << " ms total, "
                        << (s.total_ms / s.count) << " ms avg\n";
        }
    }

    void KernelProfiler::writeCsv(const std::string &path) const {
        std::ofstream out(path);
        if (!out) {
            std::cerr << "KernelProfiler: failed to open " << path << " for writing\n";
        }

        out << "kernel_name,calls,total_ms,avg_ms\n";
        for (const auto &s : aggregate()) {
            out << s.kernel_name << "," << s.count << "," << s.total_ms << "," << s.avg_ms() << "\n";
        }
    }
} // namespace ops_mlir
