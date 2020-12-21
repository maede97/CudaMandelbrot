#pragma once

#include <string>
#include <stdexcept>

#define CM_NAMESPACE_BEGIN namespace CudaMandel {
#define CM_NAMESPACE_END }

#define MAKE_ERROR(A, B) throw std::runtime_error(std::string(A) + " " + B)

#ifndef SHADER_FOLDER
#pragma message "SHADER Folder not set!"
#define SHADER_FOLDER "__TEMP__"
#endif