cmake_minimum_required(VERSION 3.10)
project(ProjectD)

# Set C++ standard
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVK_PROTOTYPES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_PROTOTYPES -static-libgcc -static-libstdc++")

    if(WIN32)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-Bstatic,--whole-archive -lpthread -Wl,--no-whole-archive")
    endif()

# Define platform-specific compiler flags
    if(WIN32)
        add_definitions(-DVK_USE_PLATFORM_WIN32_KHR)
    elseif(UNIX)
        add_definitions(-DVK_USE_PLATFORM_XCB_KHR)
    endif()

# Force debug
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
    endif()

# Disable layers
if (CMAKE_BUILD_TYPE STREQUAL "Release" OR WIN32)
    add_compile_definitions(DISABLE_VALIDATION_LAYERS)
endif()

# Release mode
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "Compiling in Release mode")
        add_compile_definitions(NDEBUG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
        if(UNIX)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=array-bounds")
        endif()
    endif()

# Vulkan
    if(UNIX)
        find_package(Vulkan REQUIRED)
    elseif(WIN32)
        # Set the path to the Windows Vulkan SDK
        set(VULKAN_SDK_PATH "./lib/vulkan-sdk-win")
        # Include directories
        include_directories(${VULKAN_SDK_PATH}/Include)
        # Link directories
        link_directories(${VULKAN_SDK_PATH}/Lib)
    endif()

# GLFW setup
    option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
    option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
    option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
    option(GLFW_INSTALL "Generate installation target" OFF)
    option(GLFW_DOCUMENT_INTERNALS "Include internals in documentation" OFF)
    add_subdirectory(lib/glfw)
    # add_subdirectory(${GLFW_DIR} binary_dir EXCLUDE_FROM_ALL)
    include_directories(lib/glfw/include)

#* Dear ImGui setup
    set(IMGUI_DIR lib/Dear-ImGui)
    include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends ..)

    set(IMGUI_SOURCES
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
    )

# GLM setup
    if(WIN32)
        # When cross-compiling for Windows, use vendored GLM
        set(GLM_PATH "${CMAKE_SOURCE_DIR}/lib/glm/glm-1.0.1")
        include_directories(${GLM_PATH})
    else()
        # On Linux host, use system GLM
        find_package(glm REQUIRED)
    endif()

    # glslangValidator setup
    find_program(GLSLANG_VALIDATOR glslangValidator
        PATHS
            ${VULKAN_SDK_PATH}/Bin
            /usr/bin
            /usr/local/bin
    )
    if(NOT GLSLANG_VALIDATOR)
        message(FATAL_ERROR "glslangValidator not found!")
    endif()
    message(STATUS "glslangValidator found: ${GLSLANG_VALIDATOR}")

# Shader compilation
set(SHADER_DIR ${CMAKE_SOURCE_DIR}/src/client/assets/shaders)
set(COMPILED_SHADER_DIR ${CMAKE_BINARY_DIR}/shaders)

file(MAKE_DIRECTORY ${COMPILED_SHADER_DIR})

set(SHADERS
    ${SHADER_DIR}/vertex.glsl
    ${SHADER_DIR}/fragment.glsl
)

# stb_image
set(STB_DIR ${CMAKE_SOURCE_DIR}/lib/stb)
include_directories(${STB_DIR})
set(STB_SOURCES src/client/image/stb_image_impl.cpp)

# assimp
add_subdirectory(lib/assimp)
include_directories(lib/assimp/include)

# Shader compilation
    set(SHADER_DIR ${CMAKE_SOURCE_DIR}/src/client/assets/shaders)
    set(COMPILED_SHADER_DIR ${CMAKE_BINARY_DIR}/shaders)

    file(MAKE_DIRECTORY ${COMPILED_SHADER_DIR})

    set(SHADERS
        vertex.glsl
        fragment.glsl
    )

    set(SHADER_OUTPUTS "")

    foreach(SHADER_NAME IN LISTS SHADERS)
        set(SHADER ${SHADER_DIR}/${SHADER_NAME})
        set(SPV_FILE ${COMPILED_SHADER_DIR}/${SHADER_NAME}.spv)

        # Detect stage based on filename
        if(SHADER_NAME MATCHES "vertex.glsl")
            set(STAGE "vert")
        elseif(SHADER_NAME MATCHES "fragment.glsl")
            set(STAGE "frag")
        else()
            message(FATAL_ERROR "Unknown shader stage for ${SHADER_NAME}")
        endif()

        add_custom_command(
            OUTPUT ${SPV_FILE}
            COMMAND ${GLSLANG_VALIDATOR} -V -S ${STAGE} ${SHADER} -o ${SPV_FILE}
            DEPENDS ${SHADER}
            COMMENT "Compiling ${SHADER_NAME} to SPIR-V with stage ${STAGE}"
            VERBATIM
        )

        list(APPEND SHADER_OUTPUTS ${SPV_FILE})
    endforeach()

add_custom_target(Shaders ALL DEPENDS ${SHADER_OUTPUTS})

# Libraries
    if(WIN32)
        set(LIBRARIES glfw assimp)
    elseif(UNIX)
        set(LIBRARIES glfw Vulkan::Vulkan pthread dl X11 Xrandr Xi Xcursor glm::glm assimp)
    endif()

    # Use Vulkan headers from GLFW
    include_directories(${GLFW_DIR}/deps)

# Add source files
    file(GLOB_RECURSE SOURCES src/*.cpp)

# Create the executable
    add_executable(ProjectD ${SOURCES} ${IMGUI_SOURCES} ${STB_SOURCES})
    add_dependencies(ProjectD Shaders)

    if(WIN32)
        set_target_properties(
            ProjectD
            PROPERTIES
                OUTPUT_NAME "ProjectD"
                SUFFIX ".exe"
        )
    elseif(UNIX)
        set_target_properties(
            ProjectD
            PROPERTIES
                OUTPUT_NAME "ProjectD"
                SUFFIX ".out"
        )

        # embedded libs
        set(LIBS_DIR "${CMAKE_SOURCE_DIR}/lib/linux")

        file(GLOB EXTRA_LIBS "${LIBS_DIR}/*.so*")
        file(COPY ${EXTRA_LIBS} DESTINATION ${CMAKE_BINARY_DIR}/lib/linux)

        file(COPY ${LIBS_DIR}/libdecor DESTINATION ${CMAKE_BINARY_DIR}/lib/linux)

        set_target_properties(ProjectD PROPERTIES
            INSTALL_RPATH "$ORIGIN/lib/linux"
            BUILD_WITH_INSTALL_RPATH TRUE
        )

    endif()

# Link libraries
if(WIN32)
    # Link against the Vulkan loader library
    target_link_libraries(ProjectD vulkan-1 -static-libgcc -static-libstdc++)
endif()

target_link_libraries(ProjectD glfw)
target_link_libraries(ProjectD ${LIBRARIES})