#include "application.hpp"

#include <imgui/imgui/backends/imgui_impl_glfw.h>
#include <imgui/imgui/backends/imgui_impl_opengl3.h>

namespace core
{
    application::application(const std::string& title, uint32_t width, uint32_t height)
    {
        window = std::make_unique<window_t>(title, width, height);
        imguiSetup();
    }

    application::~application()
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();

        ImGui::DestroyContext();
    }

    void application::imguiSetup()
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io; 
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window->window(), true);
        ImGui_ImplOpenGL3_Init("#version 460");
    }

    void application::imguiNewFrame()
    {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void application::imguiEndFrame()
    {
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    void application::run()
    {
        while (!window->should_close())
        {
            glClear(GL_COLOR_BUFFER_BIT);

            imguiNewFrame();
            
            ImGui::ShowDemoWindow();
            
            imguiEndFrame();

            glfwSwapBuffers(window->window());
            
            glfwPollEvents();
        }
    }
} // namespace core
