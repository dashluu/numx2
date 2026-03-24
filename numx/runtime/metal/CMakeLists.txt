set(CMAKE_CXX_STANDARD 23)

add_subdirectory(kernels)
file(GLOB LIB_HEADER CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB LIB_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
set(MTL_CPP "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp")
add_library(metal STATIC ${LIB_HEADER} ${LIB_SRC})
target_include_directories(metal PUBLIC ${MTL_CPP})
target_link_libraries(metal PUBLIC "-framework Metal" "-framework Foundation")
# Ensure kernels are built before executable
add_dependencies(metal mtl_kernels)
