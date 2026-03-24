set(CMAKE_CXX_STANDARD 23)

if(APPLE)
    add_subdirectory(metal)
endif()

file(GLOB LIB_HEADER CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB LIB_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

add_library(runtime STATIC ${LIB_HEADER} ${LIB_SRC})
target_link_libraries(runtime PUBLIC graph)
target_link_libraries(runtime PUBLIC metal)
target_include_directories(runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
