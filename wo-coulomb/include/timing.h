#ifndef WO_COULOMB_TIMING_H
#define WO_COULOMB_TIMING_H

#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <sys/resource.h>
#endif

namespace wo_timing {

inline bool timingEnabledFromEnv(bool defaultValue = false)
{
    const char* raw = std::getenv("WO_TIMING");
    if (!raw) return defaultValue;
    std::string v(raw);
    for (auto& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    return defaultValue;
}

inline long currentRssKb()
{
#ifdef __linux__
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return -1;
    return usage.ru_maxrss;
#else
    return -1;
#endif
}

struct TimingStats {
    double total_seconds = 0.0;
    std::size_t count = 0;
    long max_rss_kb = -1;
    long max_rss_delta_kb = 0;
};

class TimingRegistry {
public:
    void addSample(const std::string& name, double seconds, long rss_before_kb, long rss_after_kb)
    {
        auto& stats = stats_[name];
        if (stats.count == 0) {
            order_.push_back(name);
        }
        stats.total_seconds += seconds;
        stats.count += 1;
        if (rss_after_kb >= 0) {
            stats.max_rss_kb = std::max(stats.max_rss_kb, rss_after_kb);
            long delta = rss_after_kb - rss_before_kb;
            stats.max_rss_delta_kb = std::max(stats.max_rss_delta_kb, delta);
        }
    }

    void printSummary(std::ostream& os) const
    {
        os << "\nTiming summary (rank 0, wall seconds)\n";
        os << std::left << std::setw(32) << "Section"
           << std::right << std::setw(10) << "Count"
           << std::setw(16) << "Total(s)"
           << std::setw(16) << "Avg(s)"
           << std::setw(16) << "MaxRSS(MB)"
           << std::setw(16) << "MaxDelta(MB)"
           << "\n";

        for (const auto& name : order_) {
            const auto& stats = stats_.at(name);
            double avg = stats.count ? stats.total_seconds / static_cast<double>(stats.count) : 0.0;
            double max_rss_mb = stats.max_rss_kb >= 0 ? static_cast<double>(stats.max_rss_kb) / 1024.0 : -1.0;
            double max_delta_mb = static_cast<double>(stats.max_rss_delta_kb) / 1024.0;
            os << std::left << std::setw(32) << name
               << std::right << std::setw(10) << stats.count
               << std::setw(16) << std::fixed << std::setprecision(4) << stats.total_seconds
               << std::setw(16) << std::fixed << std::setprecision(4) << avg;
            if (max_rss_mb >= 0) {
                os << std::setw(16) << std::fixed << std::setprecision(2) << max_rss_mb
                   << std::setw(16) << std::fixed << std::setprecision(2) << max_delta_mb;
            } else {
                os << std::setw(16) << "n/a" << std::setw(16) << "n/a";
            }
            os << "\n";
        }
        os << std::flush;
    }

private:
    std::unordered_map<std::string, TimingStats> stats_{};
    std::vector<std::string> order_{};
};

class ScopedTimer {
public:
    ScopedTimer(TimingRegistry& registry, std::string name, bool enabled)
        : registry_(&registry), name_(std::move(name)), enabled_(enabled)
    {
        if (!enabled_) return;
        start_ = std::chrono::steady_clock::now();
        rss_before_kb_ = currentRssKb();
    }

    ~ScopedTimer()
    {
        if (!enabled_) return;
        auto stop = std::chrono::steady_clock::now();
        auto seconds = std::chrono::duration<double>(stop - start_).count();
        registry_->addSample(name_, seconds, rss_before_kb_, currentRssKb());
    }

private:
    TimingRegistry* registry_;
    std::string name_;
    bool enabled_;
    std::chrono::steady_clock::time_point start_{};
    long rss_before_kb_ = -1;
};

}  // namespace wo_timing

#endif  // WO_COULOMB_TIMING_H
