#ifndef WO_COULOMB_DETERMINISM_H
#define WO_COULOMB_DETERMINISM_H

#include <cstdlib>
#include <cctype>
#include <optional>
#include <string>
#include <algorithm>
#include <omp.h>

namespace wo_determinism {

inline bool parseBoolString(const std::string& value, bool defaultValue)
{
    std::string v = value;
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    return defaultValue;
}

inline std::optional<bool> readEnvBool(const char* name)
{
    const char* raw = std::getenv(name);
    if (!raw) return std::nullopt;
    return parseBoolString(std::string(raw), false);
}

inline bool envBoolOrDefault(const char* name, bool defaultValue)
{
    auto maybe = readEnvBool(name);
    if (maybe.has_value()) return maybe.value();
    return defaultValue;
}

inline void applyDeterministicOpenMP(bool enabled)
{
    if (!enabled) return;
    omp_set_dynamic(0);
    omp_set_schedule(omp_sched_static, 0);
}

}  // namespace wo_determinism

#endif  // WO_COULOMB_DETERMINISM_H
