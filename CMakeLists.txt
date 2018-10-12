cmake_minimum_required(VERSION 3.8)
project(thesis LANGUAGES CXX CUDA)

set(OpenGL_GL_PREFERENCE GLVND)

find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-I/opt/cuda/include")
set(SOURCE_FILES_DIR src)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOMOC ON)

add_subdirectory(glm)
add_subdirectory(src/cuda)

include_directories("src")
include_directories("src/cuda")
include_directories("src/rendering")
include_directories("src/ui")
include_directories("glm")

set(SOURCES
	"src/main.cpp"
	"src/ui/mainwindow.h"
	"src/ui/mainwindow.cpp"
	"src/ui/view.h"
	"src/ui/view.cpp"
	"src/rendering/renderer.h"
	"src/rendering/renderer.cpp"
	"src/rendering/camera.h"
	"src/rendering/camera.cpp"
	"src/rendering/orbitingcamera.h"
	"src/rendering/orbitingcamera.cpp"
	"src/rendering/actioncamera.h"
	"src/rendering/actioncamera.cpp"
	"src/debugprinting.h"
	"src/particleapp.h"
	"src/particleapp.cpp"
	"src/particlesystem.h"
	"src/particlesystem.cpp"
	"src/sdf.h"
	"src/sdf.cpp"	
)

set(UI_SOURCES
	"src/ui/mainwindow.ui"
)

add_compile_definitions(CUDA)

qt5_add_resources("resources.qrc")
qt5_wrap_ui(UI_GENERATED_HEADERS ${UI_SOURCES})

add_executable(thesis ${SOURCES} ${UI_GENERATED_HEADERS})
target_link_libraries(thesis cuda GLEW::GLEW OpenGL::GL)
qt5_use_modules(thesis Widgets OpenGL Core Gui)
