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
  typename TInputImage::Pointer blurred = TInputImage::New();
  float sigma = 2.0f;
  ITKHelpers::BlurAllChannels(image, blurred.GetPointer(), sigma);

  itk::ImageRegionConstIterator<TInputImage> imageIterator(image,image->GetLargestPossibleRegion());

  while(!imageIterator.IsAtEnd())
    {
    itk::ImageRegion<2> region = ITKHelpers::GetRegionInRadiusAroundPixel(imageIterator.GetIndex(), patchRadius);
    if(image->GetLargestPossibleRegion().IsInside(region))
      {
      itk::ImageRegionConstIterator<TInputImage> region1Iterator(image, region);
      itk::ImageRegionConstIterator<TInputImage> region2Iterator(blurred, region);
      float sum = 0.0f;
      while(!imageIterator.IsAtEnd())
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

#endif
