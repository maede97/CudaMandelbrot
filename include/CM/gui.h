#include <CM/utils.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <CM/gpu.h>
#include <CM/shader.h>
#include <CM/timer.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

CM_NAMESPACE_BEGIN

class Application
{
public:
    Application(int width = 100, int height = 100, const char* title = "CudaMandelbrot");

    ~Application();

    /**
     * Run this app
     */
    void run();

    void updateSize(int width, int height);

private:
    static void error_callback(int error, const char* description);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void resize_callback(GLFWwindow* window, int width, int height);
    static void button_callback(GLFWwindow* window, int button, int action, int mods);

    // GLFW stuff
    GLFWwindow* window;
    int width, height;

    GLuint textureID; // the texture to be displayed
    GLuint VAO;
    GLuint VBO;
    GLuint EBO;

    GLuint m_pbo = 0;
    cudaGraphicsResource* cuda_gfx_res = nullptr;

    Shader* shader = nullptr;

    unsigned int frame = 0;

    Timer fpsTimer;

    Helper helper;
};

CM_NAMESPACE_END