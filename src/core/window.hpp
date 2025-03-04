#ifndef CORE_WINDOW_HPP
#define CORE_WINDOW_HPP

#include <cstdint>
#include <string>

#include <glfw/glfw3.h>

namespace core
{
    class window_t
    {
    public:
        window_t(const std::string &title, uint32_t width, uint32_t height);
        ~window_t();

        std::string title() const { return _title; }
        GLFWwindow *window() const { return _p_window; }
        bool should_close() const { return glfwWindowShouldClose(_p_window); }
        std::pair<int, int> dimensions() const 
        {
            int width, height;

            glfwGetWindowSize(_p_window, &width, &height);

            return std::pair<int, int>(width, height); 
        }
      
        operator GLFWwindow *() const { return _p_window; }      
    private:
        std::string _title;
        GLFWwindow *_p_window;
    };
} // namespace INNER_SYSTEM

#endif // CORE_WINDOW_HPP