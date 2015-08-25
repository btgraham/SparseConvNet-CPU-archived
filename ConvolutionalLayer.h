#pragma once
#include "SpatiallySparseLayer.h"

class ConvolutionalLayer : public SpatiallySparseLayer {
private:
  int fs;
public:
  int inSpatialSize;
  int outSpatialSize;
  int filterSize;
  int filterStride;
  int dimension;
  int nFeaturesIn;
  int nFeaturesOut;
  int minActiveInputs;
  ConvolutionalLayer(int filterSize,
                     int filterStride,
                     int dimension,
                     int nFeaturesIn,
                     int minActiveInputs=1);
  void preprocess
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output);
  void forwards
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output);
  void backwards(SpatiallySparseBatch &batch,
                 SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output,
                 float learningRate,
                 float momentum);
  int calculateInputSpatialSize(int outputSpatialSize);
};

template <typename t> void convolutionFeaturesPresent(std::vector<t>& d_src, std::vector<t>& d_dest, int nf, int nfp, int nCopies);
void propForwardToMatrixMultiply(std::vector<float>& inFeatures, std::vector<float>& outFeatures, std::vector<int>& rules, int count, int nIn);
void propBackwardFromMatrixMultiply(std::vector<float>& inDFeatures, std::vector<float>& outDFeatures, std::vector<int>& rules, int count, int nIn);
