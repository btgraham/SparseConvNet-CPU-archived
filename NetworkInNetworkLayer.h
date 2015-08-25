#pragma once
#include <fstream>
#include "SpatiallySparseLayer.h"
#include "Rng.h"

class NetworkInNetworkLayer : public SpatiallySparseLayer {
private:
  RNG rng;
public:
  vectorCUDA<float> W; //Weights
  vectorCUDA<float> MW; //momentum
  vectorCUDA<float> w; //shrunk versions
  vectorCUDA<float> dw; //For backprop
  vectorCUDA<float> B; //Weights
  vectorCUDA<float> MB; //momentum
  vectorCUDA<float> b; //shrunk versions
  vectorCUDA<float> db; //For backprop
  ActivationFunction fn;
  int nFeaturesIn;
  int nFeaturesOut;
  float dropout;
  NetworkInNetworkLayer(int nFeaturesIn, int nFeaturesOut,
                        float dropout=0,ActivationFunction fn=NOSIGMOID,
                        float alpha=1//used to determine intialization weights only
                        );
  void preprocess
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output);
  void forwards
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output);
  void scaleWeights
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output,
   float& scalingUnderneath,
   bool topLayer);
  void backwards
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output,
   float learningRate,
   float momentum);
  void loadWeightsFromStream(std::ifstream &f);
  void putWeightsToStream(std::ofstream &f);
  int calculateInputSpatialSize(int outputSpatialSize);
};

void dShrinkMatrixForDropout
(std::vector<float>& m, std::vector<float>& md,
 std::vector<int>& inFeaturesPresent, std::vector<int>& outFeaturesPresent,
 int nOut);
void dShrinkVectorForDropout
(std::vector<float>& m, std::vector<float>& md, std::vector<int>& outFeaturesPresent);
void dGradientDescent
(std::vector<float>& d_delta, std::vector<float>& d_momentum, std::vector<float>& d_weights,
 int nIn, int nOut, float learningRate, float momentum);
void dGradientDescentShrunkMatrix
(std::vector<float>& d_delta, std::vector<float>& d_momentum, std::vector<float>& d_weights,
 int nOut,
 std::vector<int>& inFeaturesPresent, std::vector<int>& outFeaturesPresent,
 float learningRate,float momentum);
void dGradientDescentShrunkVector
(std::vector<float>& d_delta, std::vector<float>& d_momentum, std::vector<float>& d_weights,
 std::vector<int>& outFeaturesPresent,
 float learningRate,float momentum);
void columnSum(std::vector<float>& matrix, std::vector<float>& target, int nRows, int nColumns);
