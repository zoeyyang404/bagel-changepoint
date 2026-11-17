cmake_minimum_required(VERSION 3.20...4.2)
project(bagel_python LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tools/sync_core.py")
  execute_process(COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../tools/sync_core.py" --check
    RESULT_VARIABLE REPOSITORY_CORE_CHECK)
  if(NOT REPOSITORY_CORE_CHECK EQUAL 0)
    message(FATAL_ERROR "The shared core copies are out of sync. Run tools/sync_core.py.")
  endif()
endif()

# Both dependencies are header-only for this project. They are build-time
# dependencies; importing the resulting wheel does not require R or Boost DLLs.
find_path(EIGEN3_INCLUDE_DIR Eigen/Core
  HINTS ENV EIGEN3_INCLUDE_DIR
  PATH_SUFFIXES eigen3)
find_path(BOOST_INCLUDE_DIR boost/math/distributions/normal.hpp
  HINTS ENV BOOST_INCLUDE_DIR ENV BOOST_INCLUDEDIR ENV BOOST_ROOT
  PATH_SUFFIXES include)
if(NOT EIGEN3_INCLUDE_DIR OR NOT BOOST_INCLUDE_DIR)
  message(FATAL_ERROR
    "Install Eigen and Boost headers, or set EIGEN3_INCLUDE_DIR and BOOST_INCLUDE_DIR. See README.md.")
endif()

# A source archive must contain its own checked, synchronized core.
execute_process(COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cpp/check_vendor.py"
  RESULT_VARIABLE CORE_CHECK)
if(NOT CORE_CHECK EQUAL 0)
  message(FATAL_ERROR "Vendored core does not match its manifest. Run tools/sync_core.py in the repository.")
endif()

set(CORE "${CMAKE_CURRENT_SOURCE_DIR}/vendor/bagel")
pybind11_add_module(_core NO_EXTRAS
  cpp/bindings.cpp
  "${CORE}/normal_cdf.cpp"
  "${CORE}/normal_density.cpp"
  "${CORE}/location_scale_t_density.cpp"
  "${CORE}/noise_type.cpp"
  "${CORE}/prior_type.cpp")
target_compile_features(_core PRIVATE cxx_std_20)
target_include_directories(_core PRIVATE "${CORE}")
target_include_directories(_core SYSTEM PRIVATE "${EIGEN3_INCLUDE_DIR}" "${BOOST_INCLUDE_DIR}")
if(MSVC)
  target_compile_options(_core PRIVATE /bigobj /permissive- /EHsc)
endif()
install(TARGETS _core LIBRARY DESTINATION bagel RUNTIME DESTINATION bagel)
