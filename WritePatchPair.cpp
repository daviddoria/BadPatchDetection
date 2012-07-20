#include "PatchComparison/PairReader.h"

// STL
#include <vector>

// Submodules
#include "ITKHelpers/ITKHelpers.h"

int main(int argc, char* argv[])
{
  if(argc < 5)
  {
    throw std::runtime_error("Required arguments: imageFilename matchFilename patchRadius pairId");
  }

  std::stringstream ss;
  for(int i = 1; i < argc; ++i)
  {
    ss << argv[i] << " ";
  }

  std::string imageFileName;
  std::string matchFileName;
  unsigned int patchRadius;
  unsigned int pairId;
  ss >> imageFileName >> matchFileName >> patchRadius >> pairId;

  std::cout << "Image Filename: " << imageFileName << std::endl
            << "Match Filename: " << matchFileName << std::endl
            << "patchRadius: " << patchRadius << std::endl
            << "pairId: " << pairId << std::endl;

  // Read the image
  typedef itk::VectorImage<unsigned char, 2> ImageType;
  typedef itk::ImageFileReader<ImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(imageFileName);
  reader->Update();
  ImageType* image = reader->GetOutput();

  // Read the matches
  itk::Size<2> patchSize = {{patchRadius*2 + 1, patchRadius*2 + 1}};

  std::vector<PairReader::PairType> pairs = PairReader::Read(matchFileName, patchSize);

  // Extract the match of interest
  PairReader::PairType selectedPair = pairs[pairId];

  itk::ImageRegion<2> targetRegion = selectedPair.first;

  itk::ImageRegion<2> sourceRegion = selectedPair.second;

  // Write the patches
  ITKHelpers::WriteRegion(image, targetRegion, "target.png");
  ITKHelpers::WriteRegion(image, sourceRegion, "source.png");

  return EXIT_SUCCESS;
}
