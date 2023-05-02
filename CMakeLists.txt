# CMake file for LV2 rt-neural-generic plugin

cmake_minimum_required(VERSION 3.15)

project(mod-neural-amp VERSION 1.0.0)

set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)

set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

set(RTNEURAL_EIGEN ON)
set(RTNEURAL_XSIMD OFF)

find_package(PkgConfig)
pkg_check_modules(LV2 REQUIRED lv2>=1.10.0)

add_subdirectory(aidadsp-lv2/modules/RTNeural ${CMAKE_CURRENT_BINARY_DIR}/RTNeural)

add_custom_command(
  PRE_BUILD
  COMMAND ${CMAKE_SOURCE_DIR}/res2c.py models ${CMAKE_SOURCE_DIR}/models ${CMAKE_BINARY_DIR}
  MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/models
  OUTPUT models.cpp
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_library(mod-neural-amp SHARED
    model-from-index.cpp
    models.cpp
    aidadsp-lv2/common/Biquad.cpp
    aidadsp-lv2/rt-neural-generic/src/rt-neural-generic.cpp
)

target_compile_definitions(mod-neural-amp PRIVATE
    AIDADSP_COMMERCIAL=0
    AIDADSP_CONDITIONED_MODELS=0
    AIDADSP_MODEL_LOADER=0
    PLUGIN_URI="https://mod.audio/plugins/mod-neural-amp"
)

target_include_directories(mod-neural-amp PRIVATE
    aidadsp-lv2/common
    aidadsp-lv2/modules/RTNeural
    aidadsp-lv2/modules/RTNeural/modules/json
    aidadsp-lv2/rt-neural-generic/src
    ${LV2_INCLUDE_DIRS}
)

target_link_libraries(mod-neural-amp
    ${LV2_LIBRARIES}
    RTNeural
)

target_link_options(mod-neural-amp PRIVATE
  -Wl,--no-undefined
)

set_target_properties(mod-neural-amp
    PROPERTIES
        PREFIX ""
)

set(LV2_INSTALL_DIR ${DESTDIR}/usr/lib/lv2/mod-neural-amp.lv2)

install(FILES
    mod-neural-amp.lv2/manifest.ttl
    mod-neural-amp.lv2/mod-neural-amp.ttl
    DESTINATION ${LV2_INSTALL_DIR}
)

install(TARGETS mod-neural-amp
    LIBRARY
    DESTINATION ${LV2_INSTALL_DIR}
)
