//This does not really implement a convolution. It just gathers data together in prepartion for matrix muliplictation. "Proper convolution" = "ConvolutionalLayer" + "NetworkInNetworkLayer"

#include "ConvolutionalLayer.h"
#include <iostream>
#include <vector>
#include <cassert>
#include "cudaUtilities.h"
#include "utilities.h"
#include "Regions.h"

void propForwardToMatrixMultiply(std::vector<float>& inFeatures, std::vector<float>& outFeatures, std::vector<int>& rules, int count, int nIn) {
  for (int row=0; row<count;row++) {
    int r=rules[row];
    for (int i=0;i<nIn;i++) {
      outFeatures[row*nIn+i]=inFeatures[r*nIn+i];
    }
  }
}
void propBackwardFromMatrixMultiply(std::vector<float>& inDFeatures, std::vector<float>& outDFeatures, std::vector<int>& rules, int count, int nIn) {
  for (int row=0; row<count;row++) {
    int r=rules[row];
    for (int i=0;i<nIn;i++) {
      inDFeatures[r*nIn+i]+=outDFeatures[row*nIn+i];
    }
  }
}

template <typename t> void convolutionFeaturesPresent(std::vector<t>& d_src, std::vector<t>& d_dest, int nf, int nfp, int nCopies) {
  for (int i=0;i<nfp*nCopies;++i) {
    d_dest[i]=d_src[i%nfp]+nf*(i/nfp);
  }
}
template void convolutionFeaturesPresent<int>(std::vector<int>& d_src, std::vector<int>& d_dest, int nf, int nfp, int nCopies);

ConvolutionalLayer::ConvolutionalLayer(int filterSize,
                                       int filterStride,
                                       int dimension,
                                       int nFeaturesIn,
                                       int minActiveInputs) :
  filterSize(filterSize),
  filterStride(filterStride),
  dimension(dimension),
  nFeaturesIn(nFeaturesIn),
  minActiveInputs(minActiveInputs) {
  fs=ipow(filterSize,dimension);
  nFeaturesOut=fs*nFeaturesIn;
  std::cout << "Convolution "
            << filterSize <<"^" <<dimension<< "x"<< nFeaturesIn
            << "->" << nFeaturesOut;
  if (filterStride>1)
    std::cout << " stride:" << filterStride;
  if (minActiveInputs>1)
    std::cout << " minActiveInputs:"  << minActiveInputs;
  std::cout << std::endl;
  }
void ConvolutionalLayer::preprocess
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  output.nFeatures=nFeaturesOut;
  assert(input.nFeatures==nFeaturesIn);
  assert(input.spatialSize>=filterSize);
  assert((input.spatialSize-filterSize)%filterStride==0);
  output.spatialSize=(input.spatialSize-filterSize)/filterStride+1;
  output.nSpatialSites=0;
  output.grids.resize(batch.batchSize);
  output.backpropErrors=input.backpropErrors;
  RegularPoolingRegions regions(inSpatialSize, outSpatialSize,dimension,filterSize, filterStride);
  for (int item=0;item<batch.batchSize;item++) {
    gridRules(input.grids[item],
              output.grids[item],
              regions,
              output.nSpatialSites,
              output.rules.vec,
              minActiveInputs);
  }
  output.featuresPresent.resize(input.featuresPresent.size()*fs);
  convolutionFeaturesPresent(input.featuresPresent.vec, output.featuresPresent.vec, input.nFeatures, input.featuresPresent.size(), fs);
}
void ConvolutionalLayer::forwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  output.sub->features.resize(output.nSpatialSites*output.featuresPresent.size());
  propForwardToMatrixMultiply(input.sub->features.vec,
                              output.sub->features.vec,
                              output.rules.vec,
                              output.nSpatialSites*fs,
                              input.featuresPresent.size());
}
void ConvolutionalLayer::backwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output,
 float learningRate,
 float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
    input.sub->dfeatures.setZero();
    propBackwardFromMatrixMultiply(input.sub->dfeatures.vec,
                                   output.sub->dfeatures.vec,
                                   output.rules.vec,
                                   output.nSpatialSites*fs,
                                   input.featuresPresent.size());
  }
}
int ConvolutionalLayer::calculateInputSpatialSize(int outputSpatialSize) {
  outSpatialSize=outputSpatialSize;
  inSpatialSize=filterSize+(outputSpatialSize-1)*filterStride;
  return inSpatialSize;
}
