#ifndef CustomFilters_H
#define CustomFilters_H

#include "itkImage.h"

template <typename TInputImage, typename TOutputImage>
void ComputeSNRImage(const TInputImage* const image, const unsigned int patchRadius,
                     TOutputImage* const output);

template <typename TInputImage, typename TOutputImage>
void ComputeBlurDifferenceImage(const TInputImage* const image, const unsigned int patchRadius,
                                TOutputImage* const output);

#include "CustomFilters.hpp"

#endif
