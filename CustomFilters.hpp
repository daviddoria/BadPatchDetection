#ifndef CustomFilters_hpp
#define CustomFilters_hpp

#include "CustomFilters.h"

#include "itkImageRegionIterator.h"

#include "PatchComparison/Mask/ITKHelpers/ITKHelpers.h"
#include "PatchComparison/Mask/ITKHelpers/ITKTypeTraits.h"

template <typename TInputImage, typename TOutputImage>
void ComputeSNRImage(const TInputImage* const image, const unsigned int patchRadius,
                     TOutputImage* const output)
{
  // Initialize output
  output->SetRegions(image->GetLargestPossibleRegion());
  output->Allocate();
  output->FillBuffer(0.0f);

  itk::ImageRegionConstIterator<TInputImage> imageIterator(image,image->GetLargestPossibleRegion());

  while(!imageIterator.IsAtEnd())
    {
    itk::ImageRegion<2> region = ITKHelpers::GetRegionInRadiusAroundPixel(imageIterator.GetIndex(), patchRadius);
    if(image->GetLargestPossibleRegion().IsInside(region))
      {
      typedef typename TypeTraits<typename TInputImage::PixelType>::LargerType MultichannelType;
      MultichannelType averageInRegion = ITKHelpers::AverageInRegion(image, region);
      MultichannelType varianceInRegion = ITKHelpers::VarianceInRegion(image, region);
      float snr = 0.0f;
      for(unsigned int i = 0; i < averageInRegion.GetNumberOfComponents(); ++i)
        {
        snr += averageInRegion[i] / sqrt(varianceInRegion[i]);
        }
      output->SetPixel(imageIterator.GetIndex(), snr);
      }

    ++imageIterator;
    }
}

template <typename TInputImage, typename TOutputImage>
void ComputeBlurDifferenceImage(const TInputImage* const image, const unsigned int patchRadius,
                                TOutputImage* const output)
{
  // Initialize output
  output->SetRegions(image->GetLargestPossibleRegion());
  output->Allocate();
  output->FillBuffer(0.0f);

  typename TInputImage::Pointer blurred = TInputImage::New();
  float sigma = 2.0f;
  ITKHelpers::BlurAllChannels(image, blurred.GetPointer(), sigma);
  ITKHelpers::WriteImage(blurred.GetPointer(), "blurred.png");

  itk::ImageRegionConstIterator<TInputImage> imageIterator(image,image->GetLargestPossibleRegion());

  while(!imageIterator.IsAtEnd())
    {
    itk::ImageRegion<2> region = ITKHelpers::GetRegionInRadiusAroundPixel(imageIterator.GetIndex(), patchRadius);
    if(image->GetLargestPossibleRegion().IsInside(region))
      {
      itk::ImageRegionConstIterator<TInputImage> region1Iterator(image, region);
      itk::ImageRegionConstIterator<TInputImage> region2Iterator(blurred, region);
      float sum = 0.0f;
      while(!region2Iterator.IsAtEnd())
        {
        sum += (region1Iterator.Get() - region2Iterator.Get()).GetNorm();
        ++region1Iterator;
        ++region2Iterator;
        }
      output->SetPixel(imageIterator.GetIndex(), sum);
      }

    ++imageIterator;
    }
}

template <typename TInputImage, typename TOutputImage>
void ComputeNeighborDifferenceImage(const TInputImage* const image, const unsigned int patchRadius,
                                    TOutputImage* const output)
{
  // Initialize output
  output->SetRegions(image->GetLargestPossibleRegion());
  output->Allocate();
  output->FillBuffer(0.0f);

  itk::ImageRegionConstIteratorWithIndex<TInputImage> imageIterator(image,image->GetLargestPossibleRegion());

  while(!imageIterator.IsAtEnd())
    {
    itk::ImageRegion<2> region = ITKHelpers::GetRegionInRadiusAroundPixel(imageIterator.GetIndex(), patchRadius);

    if(image->GetLargestPossibleRegion().IsInside(region))
      {
      std::vector<itk::Index<2> > neighborIndices = ITKHelpers::Get8NeighborsInRegion(image->GetLargestPossibleRegion(),
                                                                                      imageIterator.GetIndex());
      std::vector<float> neighborDifferences;
      for(unsigned int i = 0; i < neighborIndices.size(); ++i)
      {
        itk::ImageRegion<2> neighborRegion = ITKHelpers::GetRegionInRadiusAroundPixel(neighborIndices[i], patchRadius);
        if(image->GetLargestPossibleRegion().IsInside(neighborRegion))
        {
          itk::ImageRegionConstIterator<TInputImage> regionIterator(image, region);
          itk::ImageRegionConstIterator<TInputImage> neighborRegionIterator(image, neighborRegion);
          float sum = 0.0f;
          while(!regionIterator.IsAtEnd())
          {
            sum += (regionIterator.Get() - neighborRegionIterator.Get()).GetNorm();
            ++regionIterator;
            ++neighborRegionIterator;
          }
          neighborDifferences.push_back(sum);
        } // end if this is a valid neighbor patch
      } // end loop over neighbors

      float averageNeighborDifference = 0.0f;
      if(neighborDifferences.size() > 0)
      {
        averageNeighborDifference = Statistics::Average(neighborDifferences);
      }

      output->SetPixel(imageIterator.GetIndex(), averageNeighborDifference);
      } // end if this is a valid patch to compare

      ++imageIterator;
    } // end loop over whole image
}

#endif
