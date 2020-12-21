cmake_minimum_required(VERSION 3.8)

# CudaMandelbrot
project(CudaMandelbrot LANGUAGES CXX CUDA)

include(FetchContent)
macro(fetch what)
    FetchContent_GetProperties("${what}")
    if(NOT ${${what}_POPULATED})
        message(STATUS "fetching ${what} ...")
        FetchContent_Populate(${what})
    endif()
    mark_as_advanced(${${what}_SOURCE_DIR})
endmacro()

set(SHADER_FOLDER
    "\"${CMAKE_CURRENT_LIST_DIR}/shaders\""
    CACHE STRING ""
)

# Add source files
file(GLOB_RECURSE CXX_SOURCES src/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES src/*.cu)

enable_language("CUDA")
add_executable(CudaMandelbrot ${CXX_SOURCES} ${CUDA_SOURCES})

target_compile_definitions(CudaMandelbrot PUBLIC SHADER_FOLDER=${SHADER_FOLDER})

# Add all dependencies via external projects and ext
add_subdirectory(ext/glad)

FetchContent_Declare(
    glfw
    URL https://github.com/glfw/glfw/archive/3.3.2.zip
    URL_HASH MD5=69d0aa90a9278c0dd8be975d1ae4aaf2
    UPDATE_COMMAND "" CONFIGURE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND ""
)
fetch(glfw)

FetchContent_Declare(
    glm
    URL https://github.com/g-truc/glm/releases/download/0.9.9.8/glm-0.9.9.8.zip
    URL_HASH MD5=69895110052f0d711c9c54fbf385f6f5
    UPDATE_COMMAND "" CONFIGURE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND ""
)
fetch(glm)

find_package(Eigen3)

# build options
set(GLFW_BUILD_EXAMPLES
    OFF
    CACHE BOOL "GLFW lib only"
)
set(GLFW_BUILD_TESTS
    OFF
    CACHE BOOL "GLFW lib only"
)
set(GLFW_BUILD_DOCS
    OFF
    CACHE BOOL "GLFW lib only"
)
set(GLFW_INSTALL
    OFF
    CACHE BOOL "GLFW lib only"
)

add_subdirectory(${glfw_SOURCE_DIR} glfw)
add_subdirectory(${glm_SOURCE_DIR} glm)

add_dependencies(CudaMandelbrot glfw glm)
target_include_directories(CudaMandelbrot PUBLIC include ext/glfw glm glad Eigen3::Eigen)
target_link_libraries(CudaMandelbrot glfw glm glad Eigen3::Eigen)
