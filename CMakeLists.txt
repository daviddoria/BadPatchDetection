cmake_minimum_required(VERSION 2.6)

PROJECT(BadPatchDetection)

# Necessary for c++11 usage in Helpers subsubmodule
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")

FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

# Not sure why this is necessary here - it gets included from the PatchClustering submodule.
# The error without this line is "PatchClustering.h error: Eigen/Dense: No such file or directory"
include_directories(/media/portable/src/Eigen)

add_subdirectory(PatchClustering)
add_subdirectory(PatchComparison)

ADD_EXECUTABLE(PotentialBadMatches PotentialBadMatches.cpp)
TARGET_LINK_LIBRARIES(PotentialBadMatches ${ITK_LIBRARIES} ITKHelpers)

ADD_EXECUTABLE(PotentialBadMatchesProjected PotentialBadMatchesProjected.cpp)
TARGET_LINK_LIBRARIES(PotentialBadMatchesProjected ${ITK_LIBRARIES} ITKHelpers EigenHelpers)

ADD_EXECUTABLE(CompareBadPatches CompareBadPatches.cpp)
TARGET_LINK_LIBRARIES(CompareBadPatches ${ITK_LIBRARIES} ITKHelpers)
