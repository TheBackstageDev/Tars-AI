#include "core/application.hpp"
#include <iostream>

int main() 
{
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
