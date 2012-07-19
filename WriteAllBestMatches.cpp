// STL
#include <iostream>
#include <vector>

// Submodules
#include "Helpers/Helpers.h"
#include "ITKHelpers/ITKHelpers.h"
#include "PatchComparison/SelfPatchCompare.h"
#include "PatchComparison/SSD.h"

int main(int argc, char* argv[])
{
  // Check input arguments
  if(argc < 4)
  {
    std::cerr << "Required arguments: inputFileName patchRadius logFilename" << std::endl;
    return EXIT_FAILURE;
  }

  // Parse input arguments
  std::stringstream ss;
  for(int i = 1; i < argc; ++i)
  {
    ss << argv[i] << " ";
  }
  std::string inputFileName;
  unsigned int patchRadius;
  std::string logFileName;

  ss >> inputFileName >> patchRadius >> logFileName;

  // Output arguments
  std::cout << "Image filename: " << inputFileName << std::endl
            << "patchRadius = " << patchRadius << std::endl
            << "Log filename: " << logFileName << std::endl;

  // Read the image
  typedef itk::VectorImage<unsigned char, 2> ImageType;
  //typedef itk::VectorImage<float, 2> ImageType;
  //typedef itk::Image<itk::CovariantVector<float, 3>, 2> ImageType;

  typedef itk::ImageFileReader<ImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(inputFileName);
  reader->Update();

  ImageType* image = reader->GetOutput();

  // Get all of the patches in the image
  std::vector<itk::ImageRegion<2> > allPatches = ITKHelpers::GetAllPatches(image->GetLargestPossibleRegion(), patchRadius);

  // Open the log file for writing
  std::ofstream fout(logFileName.c_str());

  for(unsigned int queryPatchId = 0; queryPatchId < allPatches.size(); ++queryPatchId)
  {
    itk::ImageRegion<2> targetRegion = allPatches[queryPatchId];
    ImageType::Pointer targetImage = ImageType::New();
    ITKHelpers::ExtractRegion(image, targetRegion, targetImage.GetPointer());
    std::stringstream ssTargetPatch;
    ssTargetPatch << "Target_" << Helpers::ZeroPad(queryPatchId, 6) << ".png";
    ITKHelpers::WriteRGBImage(targetImage.GetPointer(), ssTargetPatch.str());

    SSD<ImageType>* ssdDistanceFunctor = new SSD<ImageType>;
    ssdDistanceFunctor->SetImage(image);

    SelfPatchCompare<ImageType> selfPatchCompare;
    selfPatchCompare.SetPatchDistanceFunctor(ssdDistanceFunctor);
    selfPatchCompare.SetImage(image);
    //this->PatchCompare.SetMask(this->MaskImage);
    selfPatchCompare.CreateFullyValidMask();
    selfPatchCompare.SetTargetRegion(targetRegion);
    selfPatchCompare.ComputePatchScores();

    std::vector<SelfPatchCompare<ImageType>::PatchDataType> topPatches = selfPatchCompare.GetPatchData();

    // The very best patch should be exactly the query patch (with score 0)

    // Sort the top of the data
    std::partial_sort(topPatches.begin(), topPatches.begin() + 10, // This 10 is arbitrary
                      topPatches.end(), Helpers::SortBySecondAccending<SelfPatchCompare<ImageType>::PatchDataType>);

    if(topPatches[0].second != 0.0f)
    {
      std::stringstream ss;
      ss << "Best match should be itself (0 distance), but distance is " << topPatches[0].second;
      throw std::runtime_error(ss.str());
    }

    std::cout << "target: " << targetRegion.GetIndex() << std::endl;
    for(unsigned int i = 0; i < 10; ++i)
    {
      std::cout << "source: " << topPatches[i].first.GetIndex() << " score: " << topPatches[i].second << std::endl;
    }

    ImageType::Pointer sourceImage = ImageType::New();
    itk::ImageRegion<2> sourceRegion = topPatches[1].first; // Use the second (index=1) patch
    ITKHelpers::ExtractRegion(image, sourceRegion, sourceImage.GetPointer());
    std::stringstream ssSourcePatch;
    ssSourcePatch << "Source_" << Helpers::ZeroPad(queryPatchId, 6) << ".png";
    ITKHelpers::WriteRGBImage(sourceImage.GetPointer(), ssSourcePatch.str());

    fout << "target: " << targetRegion.GetIndex() << " source: " << sourceRegion.GetIndex()<< std::endl;
  }

  fout.close();
  return EXIT_SUCCESS;
}
