#include <iostream>
#include <vector>

#include "PatchComparison/SSD.h"

// Submodules
#include "Helpers/Helpers.h"
#include "ITKHelpers/ITKHelpers.h"
#include "PatchComparison/SelfPatchCompare.h"

int main(int argc, char* argv[])
{
  if(argc < 3)
  {
    std::cerr << "Required arguments: inputFileName patchRadius" << std::endl;
    return EXIT_FAILURE;
  }
  std::stringstream ss;
  ss << argv[1] << " " << argv[2];
  std::string inputFileName;
  unsigned int patchRadius;

  ss >> inputFileName >> patchRadius;

  std::cout << "Running on " << inputFileName << " with patchRadius = " << patchRadius << std::endl;

  typedef itk::VectorImage<float, 2> ImageType;
  //typedef itk::Image<itk::CovariantVector<float, 3>, 2> ImageType;

  typedef itk::ImageFileReader<ImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(inputFileName);
  reader->Update();

  ImageType* image = reader->GetOutput();

  std::vector<itk::ImageRegion<2> > allPatches = ITKHelpers::GetAllPatches(image->GetLargestPossibleRegion(), patchRadius);

  for(unsigned int queryPatchId = 0; queryPatchId < allPatches.size(); ++queryPatchId)
  {
    SelfPatchCompare<ImageType> selfPatchCompare;
    selfPatchCompare.SetImage(image);
    //this->PatchCompare.SetMask(this->MaskImage);
    selfPatchCompare.CreateFullyValidMask();
    selfPatchCompare.SetTargetRegion(allPatches[queryPatchId]);
    selfPatchCompare.ComputePatchScores();

    std::vector<SelfPatchCompare<ImageType>::PatchDataType> topPatches = selfPatchCompare.GetPatchData();

    // Get the top two patches, as the very best patch should be exactly the query patch (with score 0)
    std::partial_sort(topPatches.begin(), topPatches.begin() + 2,
                      topPatches.end(), Helpers::SortBySecondAccending<SelfPatchCompare<ImageType>::PatchDataType>);

    ImageType::Pointer sourceImage = ImageType::New();
    ITKHelpers::ExtractRegion(image, allPatches[queryPatchId], sourceImage.GetPointer());
    std::stringstream ssSourcePatch;
    ssSourcePatch << "Source_" << patchRadius << ".png";
    ITKHelpers::WriteImage(sourceImage.GetPointer(), ssSourcePatch.str());

    ImageType::Pointer targetImage = ImageType::New();
    ITKHelpers::ExtractRegion(image, allPatches[queryPatchId], targetImage.GetPointer());
    std::stringstream ssTargetPatch;
    ssTargetPatch << "Target_" << patchRadius << ".png";
    ITKHelpers::WriteImage(targetImage.GetPointer(), ssTargetPatch.str());

  }

  return EXIT_SUCCESS;
}
