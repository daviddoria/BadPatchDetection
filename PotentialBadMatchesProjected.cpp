#include <iostream>
#include <vector>

#include "PatchClustering/PatchClustering.h"

// Submodules
#include "PatchComparison/Mask/ITKHelpers/ITKHelpers.h"

int main(int argc, char* argv[])
{
  if(argc < 4)
  {
    std::cerr << "Required arguments: inputFileName patchRadius stride" << std::endl;
    return EXIT_FAILURE;
  }
  std::stringstream ss;
  ss << argv[1] << " " << argv[2] << " " << argv[3];
  std::string inputFileName;
  unsigned int patchRadius;
  unsigned int stride;
  ss >> inputFileName >> patchRadius >> stride;

  std::cout << "Running on " << inputFileName << " with patchRadius = " << patchRadius << " and stride = " << stride << std::endl;

  //typedef itk::VectorImage<float, 2> ImageType;
  typedef itk::Image<itk::CovariantVector<float, 3>, 2> ImageType;

  typedef itk::ImageFileReader<ImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(inputFileName);
  reader->Update();

  ImageType* image = reader->GetOutput();

  ////////////////////////////////

  itk::ImageRegionIterator<ImageType> imageIterator(image,
                                                    image->GetLargestPossibleRegion());

  EigenHelpers::VectorOfVectors vectorizedPatches;
  std::vector<itk::ImageRegion<2> > vectorizedPatchRegions;

  while(!imageIterator.IsAtEnd())
    {
    itk::ImageRegion<2> region = ITKHelpers::GetRegionInRadiusAroundPixel(imageIterator.GetIndex(), patchRadius);

    if(image->GetLargestPossibleRegion().IsInside(region))
      {
      Eigen::VectorXf v = PatchClustering::VectorizePatch(image, region);

      vectorizedPatches.push_back(v);
      vectorizedPatchRegions.push_back(region);
      }
    ++imageIterator;
    }

  std::cout << "There are " << vectorizedPatchRegions.size() << " regions." << std::endl;

  unsigned int numberOfDimensionsToKeep = 10;
  EigenHelpers::VectorOfVectors projectedVectors = EigenHelpers::DimensionalityReduction(vectorizedPatches, numberOfDimensionsToKeep);

  /////////////////////
  itk::CovariantVector<float, 3> zeroVector;
  zeroVector.Fill(0);

  typedef itk::Image<itk::CovariantVector<float, 3>, 2> OutputImageType; // (x, y, score)

  OutputImageType::Pointer locationField = OutputImageType::New();
  locationField->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
  locationField->Allocate();
  locationField->FillBuffer(zeroVector);

  OutputImageType::Pointer offsetField = OutputImageType::New();
  offsetField->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
  offsetField->Allocate();
  zeroVector.Fill(0);
  offsetField->FillBuffer(zeroVector);

  for(unsigned int i = 0; i < projectedVectors.size(); ++i)
  {
    std::cout << i << " of " << projectedVectors.size() << std::endl;

    float minDistance = std::numeric_limits<float>::max();
    unsigned int bestId = 0;

    for(unsigned int j = 0; j < projectedVectors.size(); ++j)
    {
      //std::cout << j << " of " << allPatches.size() << std::endl;
      // Don't compare a patch to itself
      if(i == j)
      {
        continue;
      }

      Eigen::VectorXf projectedVector1 = projectedVectors[i];
      Eigen::VectorXf projectedVector2 = projectedVectors[j];

      float distance = 0.0f;
      for(unsigned int component = 0; component < image->GetNumberOfComponentsPerPixel(); ++component)
      {
        distance += (projectedVector1[component] - projectedVector2[component]) * (projectedVector1[component] - projectedVector2[component]);
      }

      if(distance < minDistance)
      {
        minDistance = distance;
        bestId = j;
      }

    } // end loop j

    // Location
    itk::CovariantVector<float, 3> locationPixel;
    locationPixel[0] = vectorizedPatchRegions[bestId].GetIndex()[0];
    locationPixel[1] = vectorizedPatchRegions[bestId].GetIndex()[1];
    locationPixel[2] = minDistance;

    locationField->SetPixel(vectorizedPatchRegions[i].GetIndex(), locationPixel);

    // Offset
    itk::Offset<2> offset = vectorizedPatchRegions[bestId].GetIndex() - vectorizedPatchRegions[i].GetIndex();

    itk::CovariantVector<float, 3> offsetPixel;
    offsetPixel[0] = offset[0];
    offsetPixel[1] = offset[1];
    offsetPixel[2] = minDistance;

    offsetField->SetPixel(vectorizedPatchRegions[i].GetIndex(), offsetPixel);
  } // end loop i

  std::stringstream ssLocation;
  ssLocation << "Location_" << patchRadius << "_" << stride << ".mha";
  ITKHelpers::WriteImage(locationField.GetPointer(), ssLocation.str());

  std::stringstream ssOffset;
  ssOffset << "Offset_" << patchRadius << "_" << stride << ".mha";
  ITKHelpers::WriteImage(offsetField.GetPointer(), ssOffset.str());

  return EXIT_SUCCESS;
}
