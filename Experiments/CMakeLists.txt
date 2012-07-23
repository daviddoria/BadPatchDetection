# So that things can be included ignoring that we are in a subdirectory
include_directories(../)

add_executable(AverageNeighborBlurredScore AverageNeighborBlurredScore.cpp)
target_link_libraries(AverageNeighborBlurredScore ITKHelpers)
