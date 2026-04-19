set(CMAKE_CXX_STANDARD 23)

# Set up nanobind
execute_process(
    COMMAND uv run which python
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE Python_EXECUTABLE)
set(DEV_MODULE Development.Module)
find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

file(GLOB LIB_HEADER CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB LIB_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

nanobind_add_module(nx ${LIB_HEADER} ${LIB_SRC})
target_link_libraries(nx PUBLIC nn)
target_link_libraries(nx PUBLIC optim)
target_include_directories(nx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

function(add_stub STUB_NAME MODULE_NAME MODULE_DIR)
    nanobind_add_stub(
        ${STUB_NAME}
        MODULE nx.${MODULE_NAME}
        OUTPUT ${MODULE_DIR}/__init__.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:nx>
        DEPENDS nx
    )
endfunction()

# Add stub files
add_stub(core_stub core core)
add_stub(nn_stub nn nn)
add_stub(optim_stub optim optim)

# Define output directory for Python module and move files after build
set(PYTHON_MODULE_DIR "${CMAKE_SOURCE_DIR}/python")
message(STATUS "PYTHON_MODULE_DIR is set to: ${PYTHON_MODULE_DIR}")
get_filename_component(PYTHON_MODULE_DIR_ABS "${PYTHON_MODULE_DIR}" ABSOLUTE)
message(STATUS "Absolute path: ${PYTHON_MODULE_DIR_ABS}")

# Add custom command to move Nanobind extension after build
add_custom_target(move_extension ALL
    COMMAND ${CMAKE_COMMAND} -E rename "$<TARGET_FILE:nx>" "${PYTHON_MODULE_DIR}/$<TARGET_FILE_NAME:nx>"
    DEPENDS core_stub nn_stub optim_stub
    COMMENT "Moving Python module and stub files"
)

function(move_module SRC DEST)
    add_custom_command(TARGET move_extension POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${PYTHON_MODULE_DIR}/nx"
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${PYTHON_MODULE_DIR}/nx/${DEST}"
        COMMAND ${CMAKE_COMMAND} -E rename "${CMAKE_CURRENT_BINARY_DIR}/${SRC}" "${PYTHON_MODULE_DIR}/nx/${DEST}"
    )
endfunction()

# Move modules
move_module(core core)
move_module(nn nn)
move_module(optim optim)
