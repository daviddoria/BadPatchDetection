#include "CustomFilters.h"
#include "PatchComparison/Mask/ITKHelpers/ITKHelpers.h"

#include <iostream>

int main(int argc, char* argv[])
{
  if(argc < 4)
  {
    std::cerr << "Required arguments: imageFileName patchRadius outputFileName" << std::endl;
    return EXIT_FAILURE;
  }

  std::stringstream ss;
  for(int i = 1; i < argc; ++i)
  {
    ss << argv[i] << " ";
  }

  std::string imageFileName;
  unsigned int patchRadius;
  std::string outputFileName;
  ss >> imageFileName >> patchRadius >> outputFileName;

  std::cout << "Arguments: " << std::endl
            << "imageFileName = " << imageFileName << std::endl
            << "patchRadius = " << patchRadius << std::endl
            << "outputFileName = " << outputFileName << std::endl;

  // Read the image
  typedef itk::Image<itk::CovariantVector<unsigned char, 3>, 2> ImageType;
  ImageType::Pointer image = ImageType::New();
  ITKHelpers::ReadImage(imageFileName, image.GetPointer());

  // SNR
  ITKHelpersTypes::FloatScalarImageType::Pointer snrImage = ITKHelpersTypes::FloatScalarImageType::New();
  ComputeSNRImage(image.GetPointer(), 7, snrImage.GetPointer());
  ITKHelpers::WriteImage(snrImage.GetPointer(), "snr.mha");

  // Blur-difference
  ITKHelpersTypes::FloatScalarImageType::Pointer blurDifferenceImage = ITKHelpersTypes::FloatScalarImageType::New();
  ComputeBlurDifferenceImage(image.GetPointer(), 7, blurDifferenceImage.GetPointer());
  ITKHelpers::WriteImage(blurDifferenceImage.GetPointer(), "blurDifference.mha");

  // Average neighbor difference
  ITKHelpersTypes::FloatScalarImageType::Pointer neighborDifferenceImage = ITKHelpersTypes::FloatScalarImageType::New();
  ComputeNeighborDifferenceImage(image.GetPointer(), 7, neighborDifferenceImage.GetPointer());
  ITKHelpers::WriteImage(neighborDifferenceImage.GetPointer(), "neighborDifference.mha");

  return 0;
}

