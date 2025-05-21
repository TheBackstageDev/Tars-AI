#include "core/application.hpp"
#include <iostream>

#include "deps/tarsmath/linear_algebra/matrix_component.hpp"
#include "deps/tarscuda/tensor_operations.hpp"
#include <chrono>
#include <omp.h>

int main() 
{
    omp_set_num_threads(omp_get_max_threads());
    core::application app{"Neural Network Controller", 1000, 800};

    try
    {
        app.run();
    }
    catch(std::exception e)
    {
        std::cerr << "An Exception Occured: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
