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

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
            std::cerr << "Failed to initialize GLAD" << std::endl;
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glViewport(0, 0, width, height);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    }

    window_t::~window_t()
    {
        glfwDestroyWindow(_p_window);
        glfwTerminate();
    }
} // namespace INNER_SYSTEM
