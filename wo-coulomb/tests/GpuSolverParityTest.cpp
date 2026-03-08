#include <backend/cpu/solver_cpu.h>
#include <backend/gpu/solver_gpu_factory.h>
#include <coulombIntegral.h>
#include <testsupport/generator.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

namespace {

struct WannMaps {
    std::map<int, WannierFunction> v;
    std::map<int, WannierFunction> c;
};

WannMaps build_maps()
{
    const double L = 8.0;
    const std::vector<int> dim{24, 24, 24};
    const std::vector<double> origin{-1.0, 0.5, 0.25};
    std::vector<std::vector<double>> lattice(3);
    lattice[0] = std::vector<double>{L, 0.0, 0.0};
    lattice[1] = std::vector<double>{0.0, L, 0.0};
    lattice[2] = std::vector<double>{0.0, 0.0, L};

    auto mesh = std::make_shared<RealMeshgrid>(dim, lattice, origin);

    WannierFunction c0 = generateGauss(
        0,
        mesh,
        lattice,
        0.75,
        origin[0] + 0.45 * L,
        origin[1] + 0.48 * L,
        origin[2] + 0.47 * L,
        1.0);

    WannierFunction v0 = generateGauss(
        0,
        mesh,
        lattice,
        0.85,
        origin[0] + 0.52 * L,
        origin[1] + 0.51 * L,
        origin[2] + 0.50 * L,
        1.0);

    createLargerSupercell(c0, std::vector<int>{2, 2, 2});
    createLargerSupercell(v0, std::vector<int>{2, 2, 2});

    WannMaps maps{};
    maps.c.insert({0, std::move(c0)});
    maps.v.insert({0, std::move(v0)});
    return maps;
}

double scaled_tol(double reference, double atol, double rtol)
{
    return atol + rtol * std::max(1.0, std::abs(reference));
}

}  // namespace

TEST(GpuSolverParity, LocalFieldEffectsMatchesCpu)
{
    WannMaps maps = build_maps();

    LocalFieldEffectsSolver cpu(maps.v, maps.c);
    auto gpu = make_local_field_effects_solver_gpu(maps.v, maps.c);

    std::vector<Integral> cpu_integrals{};
    cpu_integrals.emplace_back(0, 0, 0, 0, std::vector<int>{0, 0, 0}, std::vector<int>{0, 0, 0}, std::vector<int>{0, 0, 0});
    cpu_integrals.emplace_back(0, 0, 0, 0, std::vector<int>{0, 0, 0}, std::vector<int>{1, 0, 0}, std::vector<int>{1, 0, 0});
    cpu_integrals.emplace_back(0, 0, 0, 0, std::vector<int>{0, 0, 0}, std::vector<int>{1, 0, 0}, std::vector<int>{0, 1, 0});
    cpu_integrals.emplace_back(0, 0, 0, 0, std::vector<int>{0, 0, 0}, std::vector<int>{-1, 0, 0}, std::vector<int>{0, -1, 0});

    std::vector<Integral> gpu_integrals = cpu_integrals;
    const std::vector<std::vector<int>> expected_indexes{
        cpu_integrals[0].indexes,
        cpu_integrals[1].indexes,
        cpu_integrals[2].indexes,
        cpu_integrals[3].indexes};

    cpu.calculate(cpu_integrals, false, 1, 2);
    gpu->calculate(gpu_integrals, false, 1, 2);

    for (size_t i = 0; i < cpu_integrals.size(); ++i) {
        EXPECT_EQ(gpu_integrals[i].indexes, expected_indexes[i]) << "Index tuple changed at i=" << i;
        ASSERT_FALSE(cpu_integrals[i].isFailed()) << "CPU failed at i=" << i;
        ASSERT_FALSE(gpu_integrals[i].isFailed()) << "GPU failed at i=" << i << " msg=" << gpu_integrals[i].error_msg;
        EXPECT_NEAR(
            gpu_integrals[i].value,
            cpu_integrals[i].value,
            scaled_tol(cpu_integrals[i].value, 1e-8, 5e-5))
            << "Mismatch at integral index " << i;
    }
}

TEST(GpuSolverParity, CoulombMatchesCpu)
{
    WannMaps maps = build_maps();

    CoulombSolver cpu(maps.v, maps.c, true);
    auto gpu = make_coulomb_solver_gpu(maps.v, maps.c, true);

    std::vector<Integral> cpu_integrals{};
    cpu_integrals.emplace_back(0, 0, 0, 0, std::vector<int>{0, 0, 0}, std::vector<int>{0, 0, 0}, std::vector<int>{0, 0, 0});
    cpu_integrals.emplace_back(0, 0, 0, 0, std::vector<int>{1, 0, 0}, std::vector<int>{0, 0, 0}, std::vector<int>{0, 0, 0});
    cpu_integrals.emplace_back(0, 0, 0, 0, std::vector<int>{0, 1, 1}, std::vector<int>{0, 0, 0}, std::vector<int>{0, 0, 0});
    cpu_integrals.emplace_back(0, 0, 0, 0, std::vector<int>{1, -1, 0}, std::vector<int>{1, 0, 0}, std::vector<int>{0, 1, 0});

    std::vector<Integral> gpu_integrals = cpu_integrals;
    const std::vector<std::vector<int>> expected_indexes{
        cpu_integrals[0].indexes,
        cpu_integrals[1].indexes,
        cpu_integrals[2].indexes,
        cpu_integrals[3].indexes};

    cpu.calculate(cpu_integrals, false, 1, 2);
    gpu->calculate(gpu_integrals, false, 1, 2);

    for (size_t i = 0; i < cpu_integrals.size(); ++i) {
        EXPECT_EQ(gpu_integrals[i].indexes, expected_indexes[i]) << "Index tuple changed at i=" << i;
        ASSERT_FALSE(cpu_integrals[i].isFailed()) << "CPU failed at i=" << i;
        ASSERT_FALSE(gpu_integrals[i].isFailed()) << "GPU failed at i=" << i << " msg=" << gpu_integrals[i].error_msg;
        EXPECT_NEAR(
            gpu_integrals[i].value,
            cpu_integrals[i].value,
            scaled_tol(cpu_integrals[i].value, 2e-8, 3e-4))
            << "Mismatch at integral index " << i;
    }
}

TEST(GpuSolverParity, YukawaMatchesCpu)
{
    WannMaps maps = build_maps();

    std::map<int, double> v_mean{};
    std::map<int, double> c_mean{};
    v_mean.insert({0, 0.035});
    c_mean.insert({0, 0.028});
    const double epsilon = 11.68;
    const double screening_alpha = 1.563;

    YukawaSolver cpu(maps.v, maps.c, v_mean, c_mean, epsilon, screening_alpha, make_fftw_executor_factory());
    auto gpu = make_yukawa_solver_gpu(maps.v, maps.c, v_mean, c_mean, epsilon, screening_alpha);

    std::vector<Integral> cpu_integrals{};
    cpu_integrals.emplace_back(0, 0, 0, 0, std::vector<int>{0, 0, 0}, std::vector<int>{0, 0, 0}, std::vector<int>{0, 0, 0});
    cpu_integrals.emplace_back(0, 0, 0, 0, std::vector<int>{1, 0, 0}, std::vector<int>{0, 0, 0}, std::vector<int>{0, 0, 0});
    cpu_integrals.emplace_back(0, 0, 0, 0, std::vector<int>{0, 1, 1}, std::vector<int>{0, 0, 0}, std::vector<int>{0, 0, 0});
    cpu_integrals.emplace_back(0, 0, 0, 0, std::vector<int>{1, -1, 0}, std::vector<int>{1, 0, 0}, std::vector<int>{0, 1, 0});

    std::vector<Integral> gpu_integrals = cpu_integrals;
    const std::vector<std::vector<int>> expected_indexes{
        cpu_integrals[0].indexes,
        cpu_integrals[1].indexes,
        cpu_integrals[2].indexes,
        cpu_integrals[3].indexes};

    cpu.calculate(cpu_integrals, false, 1, 2);
    gpu->calculate(gpu_integrals, false, 1, 2);

    for (size_t i = 0; i < cpu_integrals.size(); ++i) {
        EXPECT_EQ(gpu_integrals[i].indexes, expected_indexes[i]) << "Index tuple changed at i=" << i;
        ASSERT_FALSE(cpu_integrals[i].isFailed()) << "CPU failed at i=" << i;
        ASSERT_FALSE(gpu_integrals[i].isFailed()) << "GPU failed at i=" << i << " msg=" << gpu_integrals[i].error_msg;
        EXPECT_NEAR(
            gpu_integrals[i].value,
            cpu_integrals[i].value,
            scaled_tol(cpu_integrals[i].value, 1e-8, 3e-4))
            << "Mismatch at integral index " << i;
    }
}
