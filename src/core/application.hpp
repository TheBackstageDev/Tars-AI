#ifndef CORE_APPLICATION_HPP
#define CORE_APPLICATION_HPP

#include "window.hpp"
#include <memory>

namespace core
{
    class application
    {
    public:
        application(const std::string& title, uint32_t width, uint32_t height);
        ~application();

        void run();
    private:
        void imguiSetup();
        void imguiNewFrame();
        void imguiEndFrame();

        std::unique_ptr<window_t> window;
    };
    
} // namespace core

#endif // CORE_APPLICATION_HPP