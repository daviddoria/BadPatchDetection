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

  //////////// Compute the covariance matrix from a downsampled set of patches ////////////////////

  std::vector<itk::Index<2> > downsampledIndices = ITKHelpers::GetDownsampledIndicesInRegion(image->GetLargestPossibleRegion(), stride);
  std::vector<itk::ImageRegion<2> > downsampledPatches =
         ITKHelpers::GetValidPatchesCenteredAtIndices(downsampledIndices, image->GetLargestPossibleRegion(), patchRadius);

  //std::vector<itk::ImageRegion<2> > allPatches = ITKHelpers::GetAllPatches(reader->GetOutput()->GetLargestPossibleRegion(), patchRadius);
  std::cout << "There are " << downsampledPatches.size() << " patches." << std::endl;

  EigenHelpers::VectorOfVectors vectorizedDownsampledPatches;

  for(unsigned int i = 0; i < downsampledPatches.size(); ++i)
  {
    Eigen::VectorXf v = PatchClustering::VectorizePatch(image, downsampledPatches[i]);

    vectorizedDownsampledPatches.push_back(v);
  }

  std::cout << "There are " << vectorizedDownsampledPatches.size() << " vectorizedDownsampledPatches." << std::endl;

  unsigned int numberOfDimensionsToKeep = 10;

  Eigen::MatrixXf covarianceMatrix = EigenHelpers::ConstructCovarianceMatrix(vectorizedDownsampledPatches);

  std::cout << "Done computing covariance matrix." << std::endl;

  ////////// Project all of the patches using the covariance matrix constructed from the downsampled set /////

  std::vector<itk::ImageRegion<2> > allPatches = ITKHelpers::GetAllPatches(image->GetLargestPossibleRegion(), patchRadius);

  EigenHelpers::VectorOfVectors vectorizedPatches;

  for(unsigned int i = 0; i < allPatches.size(); ++i)
  {
    Eigen::VectorXf v = PatchClustering::VectorizePatch(image, allPatches[i]);
    vectorizedPatches.push_back(v);
  }

  std::cout << "Done vectorizing patches." << std::endl;

  EigenHelpers::VectorOfVectors projectedVectors =
          EigenHelpers::DimensionalityReduction(vectorizedPatches, covarianceMatrix, numberOfDimensionsToKeep);

  std::cout << "There are " << projectedVectors.size() << " projectedVectors." << std::endl;

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
  ssLocation << "Location_" << patchRadius << "_" << stride << ".mha";
  ITKHelpers::WriteImage(locationField.GetPointer(), ssLocation.str());

  std::stringstream ssOffset;
  ssOffset << "Offset_" << patchRadius << "_" << stride << ".mha";
  ITKHelpers::WriteImage(offsetField.GetPointer(), ssOffset.str());

  return EXIT_SUCCESS;
}
