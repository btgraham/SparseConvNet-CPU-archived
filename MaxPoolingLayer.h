#pragma once
#include <iostream>
#include "SpatiallySparseLayer.h"
#include "Rng.h"

void maxPool(std::vector<float>& g1, std::vector<float>& g2, std::vector<int>& rules, int count, int sd, int nOut, std::vector<int>& d_choice);
void maxPoolBackProp(std::vector<float>& d1, std::vector<float>& d2, int count, int nOut, std::vector<int>& d_choice);

//TODO: Refactor the different pooling classes somehow


class MaxPoolingLayer : public SpatiallySparseLayer {
public:
  int inSpatialSize;
  int outSpatialSize;
  int poolSize;
  int poolStride;
  int dimension;
  int sd;
  MaxPoolingLayer(int poolSize, int poolStride, int dimension);
  void preprocess
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output);
  void forwards(SpatiallySparseBatch &batch,
                SpatiallySparseBatchInterface &input,
                SpatiallySparseBatchInterface &output);
  void backwards(SpatiallySparseBatch &batch,
                 SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output,
                 float learningRate,
                 float momentum);
  int calculateInputSpatialSize(int outputSpatialSize);
};

class PseudorandomOverlappingFractionalMaxPoolingLayer : public SpatiallySparseLayer {
  int sd;
public:
  int inSpatialSize;
  int outSpatialSize;
  float fmpShrink;
  int poolSize;
  int dimension;
  RNG rng;
  PseudorandomOverlappingFractionalMaxPoolingLayer(int poolSize, float fmpShrink, int dimension);
  void preprocess
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output);
  void forwards(SpatiallySparseBatch &batch,
                SpatiallySparseBatchInterface &input,
                SpatiallySparseBatchInterface &output);
  void backwards(SpatiallySparseBatch &batch,
                 SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output,
                 float learningRate,
                 float momentum);
  int calculateInputSpatialSize(int outputSpatialSize);
};

class RandomOverlappingFractionalMaxPoolingLayer : public SpatiallySparseLayer {
  int sd;
public:
  int inSpatialSize;
  int outSpatialSize;
  float fmpShrink;
  int poolSize;
  int dimension;
  RNG rng;
  RandomOverlappingFractionalMaxPoolingLayer(int poolSize, float fmpShrink, int dimension);
  void preprocess
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output);
  void forwards
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output);
  void backwards
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output,
   float learningRate,
   float momentum);
  int calculateInputSpatialSize(int outputSpatialSize);
};
