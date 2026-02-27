// tests.cpp
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <gtest/gtest.h>
#include "determinism.h"

// all tests:
#include "MeshgridTest.cpp"
#include "SchedulerTest.cpp"
#include "WannierfunctionTest.cpp"
#include "FourierShiftTest.cpp"
#include "XSF_controllerTest.cpp"
#include "ImplementationCoulombSolverTest.cpp"
#include "ImplementationTest.cpp"
#include "LocalFieldEffectsTest.cpp"
#include "OpticalDipoleTest.cpp"
#include "DensityTest.cpp"
#include "CorrectnessSmallMediumTest.cpp"
#include "CHGCARTest.cpp"


using namespace std;


int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    const bool deterministic = wo_determinism::envBoolOrDefault("WO_DETERMINISTIC", true);
    wo_determinism::applyDeterministicOpenMP(deterministic);

    cout << fixed;
    cout << setprecision(12);
    testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();

    MPI_Finalize();
    return ret;
}
