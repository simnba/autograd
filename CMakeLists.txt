cmake_minimum_required(VERSION 3.20)
project(AutoGrad)

set(CMAKE_CXX_STANDARD 23)

file(GLOB cppSources src/*.cpp)
add_executable(AutoGrad ${cppSources})

set_property(TARGET AutoGrad PROPERTY CXX_STANDARD 23)

#libfmt.a
target_include_directories(AutoGrad PRIVATE "../fmt/include/")



