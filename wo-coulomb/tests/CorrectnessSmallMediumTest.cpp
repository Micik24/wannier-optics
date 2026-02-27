#include <backend/cpu/solver_cpu.h>
#include "testsupport/generator.h"
#include "testsupport/tolerance.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

namespace {

struct TestConfig {
    int dim;
    double L;
    double sigma;
};

TestConfig smallConfig()
{
    return TestConfig{24, 10.0, 0.6};
}

TestConfig mediumConfig()
{
    return TestConfig{32, 12.0, 0.7};
}

void buildWannierMaps(const TestConfig& cfg,
                      std::map<int, WannierFunction>& vWannMap,
                      std::map<int, WannierFunction>& cWannMap)
{
    std::vector<int> dim{cfg.dim, cfg.dim, cfg.dim};
    std::vector<double> origin{0.0, 0.0, 0.0};
    std::vector<std::vector<double>> lattice(3);
    lattice[0] = std::vector<double>{cfg.L, 0.0, 0.0};
    lattice[1] = std::vector<double>{0.0, cfg.L, 0.0};
    lattice[2] = std::vector<double>{0.0, 0.0, cfg.L};

    auto meshgrid = std::make_shared<RealMeshgrid>(dim, lattice, origin);

    const double sigma = cfg.sigma;

    // Conduction WFs: clustered near one corner
    cWannMap.insert({0, generateGauss(0, meshgrid, lattice, sigma, cfg.L * 0.20, cfg.L * 0.20, cfg.L * 0.20, 1.0)});
    cWannMap.insert({1, generateGauss(1, meshgrid, lattice, sigma, cfg.L * 0.35, cfg.L * 0.20, cfg.L * 0.20, 1.0)});

    // Valence WFs: clustered near the opposite corner
    vWannMap.insert({0, generateGauss(0, meshgrid, lattice, sigma, cfg.L * 0.80, cfg.L * 0.80, cfg.L * 0.80, 1.0)});
    vWannMap.insert({1, generateGauss(1, meshgrid, lattice, sigma, cfg.L * 0.65, cfg.L * 0.80, cfg.L * 0.80, 1.0)});
}

std::vector<Integral> buildIntegrals()
{
    std::vector<Integral> integrals;
    const std::vector<int> zero{0, 0, 0};

    integrals.emplace_back(0, 0, 0, 0, zero, zero, zero);
    integrals.emplace_back(1, 1, 1, 1, zero, zero, zero);

    return integrals;
}

std::filesystem::path dataDir()
{
    return std::filesystem::path(__FILE__).parent_path() / "data";
}

std::vector<double> readExpected(const std::filesystem::path& path)
{
    std::ifstream file(path);
    std::vector<double> values;
    double v = 0.0;
    while (file >> v) {
        values.push_back(v);
    }
    return values;
}

void writeExpected(const std::filesystem::path& path, const std::vector<double>& values)
{
    std::ofstream file(path);
    file.setf(std::ios::fixed);
    file.precision(12);
    for (double v : values) {
        file << v << "\n";
    }
}

bool updateGolden()
{
    const char* raw = std::getenv("WO_UPDATE_GOLDEN");
    if (!raw) return false;
    return std::string(raw) == "1";
}

void runGoldenTest(const TestConfig& cfg, const std::filesystem::path& expectedPath)
{
    omp_set_dynamic(0);
    omp_set_num_threads(1);

    std::map<int, WannierFunction> vWannMap;
    std::map<int, WannierFunction> cWannMap;
    buildWannierMaps(cfg, vWannMap, cWannMap);

    CoulombSolver fast(vWannMap, cWannMap);

    auto fast_integrals = buildIntegrals();

    fast.calculate(fast_integrals, false, 1, 1);

    std::vector<double> values;
    values.reserve(fast_integrals.size());
    for (const auto& integral : fast_integrals) {
        values.push_back(integral.value);
    }

    if (updateGolden()) {
        std::filesystem::create_directories(expectedPath.parent_path());
        writeExpected(expectedPath, values);
        SUCCEED() << "Updated golden file: " << expectedPath;
        return;
    }

    auto expected = readExpected(expectedPath);
    ASSERT_EQ(expected.size(), values.size()) << "Expected file missing or size mismatch: " << expectedPath;
    for (std::size_t i = 0; i < values.size(); ++i) {
        wo_test::expectNearRelAbs(values[i], expected[i], wo_test::kGoldenTolerance);
    }
}

}  // namespace

TEST(CorrectnessSmall, CoulombSolverMatchesRealSpace)
{
    runGoldenTest(smallConfig(), dataDir() / "coulomb_small.txt");
}

TEST(CorrectnessMedium, CoulombSolverMatchesRealSpace)
{
    runGoldenTest(mediumConfig(), dataDir() / "coulomb_medium.txt");
}
