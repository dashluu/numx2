set(CMAKE_CXX_STANDARD 23)

# Set up GTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB TEST_HEADER CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)

file(GLOB TEST_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)

add_executable(numx_test ${TEST_SRC} ${TEST_HEADER} ${LIB_SRC} ${LIB_HEADER})
target_link_libraries(numx_test foundation)
target_link_libraries(numx_test GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(numx_test)
