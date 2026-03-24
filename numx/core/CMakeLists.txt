set(CMAKE_CXX_STANDARD 23)
add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")

file(GLOB LIB_HEADER CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB LIB_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

add_library(core STATIC ${LIB_HEADER} ${LIB_SRC})
target_link_libraries(core PUBLIC runtime)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
