#include "window.hpp"
#include <iostream>

namespace core
{
    window_t::window_t(const std::string &title, uint32_t width, uint32_t height)
        : _title(title)
    {
        if (!glfwInit())
        {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            exit(EXIT_FAILURE);
        }

        _p_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
        if (!_p_window)
        {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(_p_window);
        glfwSwapInterval(1); // Enable vsync
    }

    window_t::~window_t()
    {
        glfwDestroyWindow(_p_window);
        glfwTerminate();
    }
} // namespace INNER_SYSTEM
