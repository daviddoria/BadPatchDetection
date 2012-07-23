// STL
#include <vector>

//ITK
#include "itkSmoothingRecursiveGaussianImageFilter.h"

// Submodules
#include "ITKHelpers/ITKHelpers.h"
#include "PatchComparison/SSD.h"
#include "Helpers/Statistics.h"

int main(int argc, char* argv[])
{
  if(argc < 4)
  {
    throw std::runtime_error("Required arguments: imageFilename patchRadius blurVariance");
  }

  std::stringstream ss;
  for(int i = 1; i < argc; ++i)
  {
    ss << argv[i] << " ";
  }

  std::string imageFileName;
  float blurVariance;
  unsigned int patchRadius;
  ss >> imageFileName >> patchRadius >> blurVariance;

  std::cout << "Image Filename: " << imageFileName << std::endl
            << "patchRadius: " << patchRadius << std::endl
            << "blurVariance: " << blurVariance << std::endl;

  // Read the image
  typedef itk::VectorImage<unsigned char, 2> ImageType;
  typedef itk::ImageFileReader<ImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(imageFileName);
  reader->Update();
  ImageType* image = reader->GetOutput();

  typedef itk::SmoothingRecursiveGaussianImageFilter<
    ImageType, ImageType >  FilterType;
  FilterType::Pointer smoothingFilter = FilterType::New();
  smoothingFilter->SetInput(image);
  smoothingFilter->SetSigma(blurVariance);
  smoothingFilter->Update();

  ImageType* blurredImage = smoothingFilter->GetOutput();
  
  // Read the matches
  itk::Size<2> patchSize = {{patchRadius*2 + 1, patchRadius*2 + 1}};

  std::vector<itk::ImageRegion<2> > allPatches = ITKHelpers::GetAllPatches(image->GetLargestPossibleRegion(), patchRadius);

  std::vector<float> allScores(allPatches.size());

  for(unsigned int patchId = 0; patchId < allPatches.size(); ++patchId)
  {
    if(patchId % 10000 == 0)
    {
      std::cout << "Computing patch " << patchId << std::endl;
    }

    std::vector<itk::ImageRegion<2> > neighborRegions = ITKHelpers::Get8NeighborRegionsInRegion(image->GetLargestPossibleRegion(),
                                                                                                ITKHelpers::GetRegionCenter(allPatches[patchId]),
                                                                                                patchSize);

    float totalScore = 0.0f;
    for(unsigned int neighborId = 0; neighborId < neighborRegions.size(); ++neighborId)
    {
      totalScore += SSD<ImageType>::Distance(blurredImage, allPatches[patchId], neighborRegions[neighborId]);
    }

    if(neighborRegions.size() > 0)
    {
      allScores[patchId] = totalScore / static_cast<float>(neighborRegions.size());
    }
    else
    {
      allScores[patchId] = 0.0f; // or this could be std::numeric_limits<float>::max() - just trying to show "invalid"
    }
  }

  std::string outputFileName = "scores.txt";
  std::ofstream fout(outputFileName.c_str());

  for(unsigned int i = 0; i < allScores.size(); ++i)
  {
    fout << allScores[i] << std::endl;
  }
  fout.close();

  std::cout << "Average: " << Statistics::Average(allScores) << std::endl;
  std::cout << "Max: " << Helpers::max(allScores) << std::endl;

  return EXIT_SUCCESS;
}
