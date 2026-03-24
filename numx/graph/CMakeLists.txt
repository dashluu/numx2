set(CMAKE_CXX_STANDARD 23)

file(GLOB LIB_HEADER CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB LIB_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

add_library(graph STATIC ${LIB_HEADER} ${LIB_SRC})
target_link_libraries(graph PUBLIC foundation)
target_include_directories(graph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
