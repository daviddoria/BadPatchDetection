cmake_minimum_required(VERSION 2.6)

PROJECT(BadPatchDetection)

# Necessary for c++11 usage in Helpers subsubmodule
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -std=gnu++0x") #

# This makes the squaredNorm() call about twice as fast, which is very important in big loops
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")

FIND_PACKAGE(ITK REQUIRED ITKCommon ITKIOImageBase ITKIOPNG ITKDistanceMap
            ITKImageIntensity ITKImageFeature ITKMathematicalMorphology ITKBinaryMathematicalMorphology)
INCLUDE(${ITK_USE_FILE})

# Not sure why this is necessary here - it gets included from the PatchClustering submodule.
# The error without this line is "PatchClustering.h error: Eigen/Dense: No such file or directory"
include_directories(/media/portable/src/Eigen)

if(NOT TARGET PatchClustering)
  add_subdirectory(PatchClustering)
  include_directories(${PatchClustering_includes})
endif()

if(NOT TARGET PatchComparison)
  add_subdirectory(PatchComparison)
  include_directories(${PatchComparison_includes})
endif()

ADD_EXECUTABLE(PotentialBadMatches PotentialBadMatches.cpp)
TARGET_LINK_LIBRARIES(PotentialBadMatches ${ITK_LIBRARIES} ITKHelpers)

# ADD_EXECUTABLE(PotentialBadMatchesVectorized PotentialBadMatchesVectorized.cpp)
# TARGET_LINK_LIBRARIES(PotentialBadMatchesVectorized ${ITK_LIBRARIES} ITKHelpers EigenHelpers)
# 
# ADD_EXECUTABLE(PotentialBadMatchesProjected PotentialBadMatchesProjected.cpp)
# TARGET_LINK_LIBRARIES(PotentialBadMatchesProjected ${ITK_LIBRARIES} ITKHelpers EigenHelpers)
# 
# ADD_EXECUTABLE(PotentialBadMatchesProjectedColorOnly PotentialBadMatchesProjectedColorOnly.cpp)
# TARGET_LINK_LIBRARIES(PotentialBadMatchesProjectedColorOnly ${ITK_LIBRARIES} ITKHelpers EigenHelpers)

ADD_EXECUTABLE(CompareBadPatches CompareBadPatches.cpp)
TARGET_LINK_LIBRARIES(CompareBadPatches ${ITK_LIBRARIES} ITKHelpers)

ADD_EXECUTABLE(TestCustomFilters TestCustomFilters.cpp)
TARGET_LINK_LIBRARIES(TestCustomFilters ${ITK_LIBRARIES} ITKHelpers)

ADD_EXECUTABLE(WriteAllBestMatches WriteAllBestMatches.cpp)
TARGET_LINK_LIBRARIES(WriteAllBestMatches ${ITK_LIBRARIES} Mask)

ADD_EXECUTABLE(WritePatchPair WritePatchPair.cpp)
TARGET_LINK_LIBRARIES(WritePatchPair ${ITK_LIBRARIES} Mask PatchComparison)
