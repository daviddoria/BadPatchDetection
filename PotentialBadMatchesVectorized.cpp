#include <iostream>
#include <vector>

#include "PatchComparison/SSD.h"

// Submodules
#include "PatchComparison/Mask/ITKHelpers/ITKHelpers.h"
#include "PatchClustering/PatchClustering.h"
#include "PatchComparison/EigenHelpers/EigenHelpers.h"

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

  std::cout << "Running on " << inputFileName << " with patchRadius = "
            << patchRadius << std::endl;

  typedef itk::Image<itk::CovariantVector<float, 3>, 2> ImageType;

  typedef itk::ImageFileReader<ImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(inputFileName);
  reader->Update();

  ImageType* image = reader->GetOutput();

  std::vector<itk::ImageRegion<2> > allPatches =
           ITKHelpers::GetAllPatches(reader->GetOutput()->GetLargestPossibleRegion(), patchRadius);

  EigenHelpers::VectorOfVectors vectorizedPatches(allPatches.size());

  for(unsigned int i = 0; i < vectorizedPatches.size(); ++i)
  {
    vectorizedPatches[i] = PatchClustering::VectorizePatch(image, allPatches[i]);
  }

  std::cout << "There are " << vectorizedPatches.size() << " vectorizedPatches with "
            << vectorizedPatches[0].size() << " components each." << std::endl;

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

  for(unsigned int i = 0; i < vectorizedPatches.size(); ++i)
  {
    printf ("%d of %d\n", i, vectorizedPatches.size());

    float minDistance = std::numeric_limits<float>::max();
    unsigned int bestId = 0;

    for(unsigned int j = 0; j < vectorizedPatches.size(); ++j)
    {
      // Don't compare a patch to itself
      if(i == j)
      {
        continue;
      }

      //float distance = (vectorizedPatches[i] - vectorizedPatches[j]).squaredNorm();
      float distance = (vectorizedPatches[i] - vectorizedPatches[j]).norm();

      if(distance < minDistance)
      {
        minDistance = distance;
        bestId = j;
      }
    } // end loop j

    itk::Index<2> patchCenter = ITKHelpers::GetRegionCenter(allPatches[i]);
    itk::Index<2> bestMatchCenter = ITKHelpers::GetRegionCenter(allPatches[bestId]);

    // Location
    itk::CovariantVector<float, 3> locationPixel;
    locationPixel[0] = bestMatchCenter[0];
    locationPixel[1] = bestMatchCenter[1];
    locationPixel[2] = minDistance;
    locationField->SetPixel(patchCenter, locationPixel);

    // Offset
    itk::Offset<2> offset = bestMatchCenter - patchCenter;

    itk::CovariantVector<float, 3> offsetPixel;
    offsetPixel[0] = offset[0];
    offsetPixel[1] = offset[1];
    offsetPixel[2] = minDistance;

    offsetField->SetPixel(patchCenter, offsetPixel);
  } // end loop i

  std::stringstream ssLocation;
  ssLocation << "Vectorized_Location_" << patchRadius << ".mha";
  ITKHelpers::WriteImage(locationField.GetPointer(), ssLocation.str());

  std::stringstream ssOffset;
  ssOffset << "Vectorized_Offset_" << patchRadius << ".mha";
  ITKHelpers::WriteImage(offsetField.GetPointer(), ssOffset.str());

  return EXIT_SUCCESS;
}
