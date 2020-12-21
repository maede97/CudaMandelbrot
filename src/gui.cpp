#include <CM/gui.h>
#include <stdexcept>
#include <iostream>
#include <vector>

#define CUDA_CHECK(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define MOVEMENT_INC 0.01

CM_NAMESPACE_BEGIN

Application::Application(int width, int height, const char *title) : width(width), height(height)
{
    checkCuda();

    if (!glfwInit())
    {
        MAKE_ERROR("GLFW", "Could not init.");
    }

    // set callback
    glfwSetErrorCallback(Application::error_callback);

    // create window with OpenGL3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window)
    {
        MAKE_ERROR("Application", "Could not create Window.");
    }

    // make context to use OpenGL functions on this window
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        MAKE_ERROR("Application", "GLAD could not load OpenGL");
    }

    glfwSetKeyCallback(window, Application::key_callback);
    glfwSetFramebufferSizeCallback(window, Application::resize_callback);
    glfwSetMouseButtonCallback(window, Application::button_callback);

    glfwSwapInterval(1);

    glfwSetWindowUserPointer(window, this);

    glViewport(0, 0, width, height);

    // Create the texture to be displayed
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    shader = new Shader(SHADER_FOLDER "/shader.vert", SHADER_FOLDER "/shader.frag");
    float vertices[] = {
        // positions          // colors           // texture coords
        1.f, 1.f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,   // top right
        1.f, -1.f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,  // bottom right
        -1.f, -1.f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // bottom left
        -1.f, 1.f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f   // top left
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenBuffers(1, &m_pbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_pbo);
    glBufferData(GL_ARRAY_BUFFER, 3 * width * height * sizeof(unsigned char), nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0u);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_gfx_res, m_pbo, cudaGraphicsMapFlagsWriteDiscard));

    float upperX = 2.f;
    float lowerX = -2.f;
    float rangeX = upperX - lowerX;
    float lowerY = 2.f;
    float upperY = -2.f;
    float rangeY = upperY - lowerY;

    helper = Helper{upperX, lowerX, upperY, lowerY, rangeX, rangeY, width, height, 1.f, 1};
}

Application::~Application()
{
    delete shader;

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    glfwDestroyWindow(window);
    glfwTerminate();
}

void Application::run()
{
    const size_t blockSizeBytes = 3 * width * height * sizeof(unsigned char);

    unsigned char *device_pixels = nullptr;
    size_t buffer_size;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_gfx_res));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&device_pixels), &buffer_size, cuda_gfx_res));

    while (!glfwWindowShouldClose(window))
    {
        fpsTimer.start();

        glClearColor(0.f, 0.f, 0.f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        // Call GPU function
        // for now: only once
        doCalc(device_pixels, helper);
        //CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(display_data), gpu_data, blockSizeBytes, cudaMemcpyDeviceToHost));
        glBindTexture(GL_TEXTURE_2D, textureID);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 3);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        //glGenerateMipmap(GL_TEXTURE_2D);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Bind Shader
        shader->use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();

        frame++;

        //helper.lowerX += 0.001;
        //helper.upperX += 0.001;

        while (fpsTimer.elapsedMilliseconds() < 1000 / 15)
        {
        }
    }
}

void Application::button_callback(GLFWwindow *window, int button, int action, int mods)
{
    Application *app = static_cast<Application *>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        double mX, mY;
        glfwGetCursorPos(window, &mX, &mY);

        // map mouse pointer to plane
        mX -= app->width / 2;
        mY = app->height / 2 - mY;

        double sum = mX * mX + mY * mY;
        mX /= std::sqrt(sum);
        mY /= std::sqrt(sum);
        app->helper.V_x = mX;
        app->helper.V_y = mY;
    }
}

void Application::error_callback(int error, const char *description)
{
    MAKE_ERROR("Application::Error", description);
}

void Application::key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    Application *app = static_cast<Application *>(glfwGetWindowUserPointer(window));
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    else if (key == GLFW_KEY_UP)
    {
        app->helper.lowerY -= MOVEMENT_INC * app->helper.zoomLevel;
        app->helper.upperY -= MOVEMENT_INC * app->helper.zoomLevel;
    }
    else if (key == GLFW_KEY_DOWN)
    {
        app->helper.lowerY += MOVEMENT_INC * app->helper.zoomLevel;
        app->helper.upperY += MOVEMENT_INC * app->helper.zoomLevel;
    }
    else if (key == GLFW_KEY_LEFT)
    {
        app->helper.lowerX -= MOVEMENT_INC * app->helper.zoomLevel;
        app->helper.upperX -= MOVEMENT_INC * app->helper.zoomLevel;
    }
    else if (key == GLFW_KEY_RIGHT)
    {
        app->helper.lowerX += MOVEMENT_INC * app->helper.zoomLevel;
        app->helper.upperX += MOVEMENT_INC * app->helper.zoomLevel;
    }
    else if (key == GLFW_KEY_E)
    {
        app->helper.lowerX += MOVEMENT_INC * app->helper.zoomLevel;
        app->helper.upperX -= MOVEMENT_INC * app->helper.zoomLevel;

        double temp = (app->helper.upperX - app->helper.lowerX) / app->helper.rangeX;
        app->helper.rangeX = app->helper.upperX - app->helper.lowerX;
        app->helper.lowerY -= MOVEMENT_INC * app->helper.zoomLevel;
        app->helper.upperY += MOVEMENT_INC * app->helper.zoomLevel;
        app->helper.rangeY = app->helper.upperY - app->helper.lowerY;
        app->helper.zoomLevel *= temp;
    }
    else if (key == GLFW_KEY_Q)
    {
        app->helper.lowerX -= MOVEMENT_INC * app->helper.zoomLevel;
        app->helper.upperX += MOVEMENT_INC * app->helper.zoomLevel;
        double temp = (app->helper.upperX - app->helper.lowerX) / app->helper.rangeX;
        app->helper.rangeX = app->helper.upperX - app->helper.lowerX;
        app->helper.lowerY += MOVEMENT_INC * app->helper.zoomLevel;
        app->helper.upperY -= MOVEMENT_INC * app->helper.zoomLevel;
        app->helper.rangeY = app->helper.upperY - app->helper.lowerY;
        app->helper.zoomLevel *= temp;
    }
    else if (key == GLFW_KEY_1)
    {
        app->helper.color = 1;
    }
    else if (key == GLFW_KEY_2)
    {
        app->helper.color = 2;
    }
    else if (key == GLFW_KEY_3)
    {
        app->helper.color = 3;
    }
    else if (key == GLFW_KEY_4)
    {
        app->helper.color = 4;
    }
    else if (key == GLFW_KEY_5)
    {
        app->helper.color = 5;
    }
}

void Application::updateSize(int width, int height)
{
    this->width = width;
    this->height = height;
}

void Application::resize_callback(GLFWwindow *window, int width, int height)
{
    Application *app = static_cast<Application *>(glfwGetWindowUserPointer(window));
    //glViewport(0, 0, width, height);
    //app->updateSize(width, height);
    // TODO: resize data arrays properly
}

CM_NAMESPACE_END