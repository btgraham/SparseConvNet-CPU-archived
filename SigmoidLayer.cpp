#include <iostream>
#include <cmath>
#include "SigmoidLayer.h"
#include "cudaUtilities.h"

void sigmoidReLU(std::vector<float>& a, std::vector<float>& b) {
  for (int i=0;i<a.size();i++)
    b[i]=(a[i]>0)?a[i]:0;
}

void sigmoidBackpropReLU
(std::vector<float>& a, std::vector<float>& b, std::vector<float>& da, std::vector<float>& db) {
  for (int i=0;i<a.size();i++)
    da[i]=(a[i]>0)?db[i]:0;
}

void sigmoidTanh(std::vector<float>& a, std::vector<float>& b) {
  for (int i=0;i<a.size();i++)
    b[i]=tanh(a[i]);
}
void sigmoidBackpropTanh
(std::vector<float>& a, std::vector<float>& b, std::vector<float>& da, std::vector<float>& db) {
  for (int i=0;i<a.size();i++)
    da[i]=db[i]*(1+b[i])*(1-b[i]);
}


void sigmoidLeakyReLU(std::vector<float>& a, std::vector<float>& b, float alpha) {
  for (int i=0;i<a.size();i++)
    b[i]=(a[i]>0)?a[i]:(a[i]*alpha);
}

void sigmoidBackpropLeakyReLU
(std::vector<float>& a, std::vector<float>& b, std::vector<float>& da, std::vector<float>& db, float alpha) {
  for (int i=0;i<a.size();i++)
    da[i]=(a[i]>0)?db[i]:(db[i]*alpha);
}

//SOFTMAX should only be used in the top layer;
//derivative contained in calculation of initial d_delta.
void sigmoidSoftmax(std::vector<float>& a, std::vector<float>& b, int nOut) {
  for (int i=0;i<a.size();i+=nOut) {
    float mx=a[i];
    for (int j=i+1;j<i+nOut;j++)
      mx=std::max(mx,a[j]);
    float acc=0;
    for (int j=i;j<i+nOut;j++)
      acc+=b[j]=expf(a[j]-mx);
    for (int j=i;j<i+nOut;j++)
      b[j]/=acc;
  }
}

void sigmoidBackpropSoftmax
(std::vector<float>& a, std::vector<float>& b, std::vector<float>& da, std::vector<float>& db, float alpha) {
  for (int i=0;i<a.size();i++)
    da[i]=db[i];
}

void applySigmoid(SpatiallySparseBatchInterface& input, SpatiallySparseBatchInterface& output, ActivationFunction fn) {
  switch(fn) {
  case TANH:
    sigmoidTanh
      (input.sub->features.vec,
       output.sub->features.vec);
    break;
  case RELU:
    sigmoidReLU
      (input.sub->features.vec,
       output.sub->features.vec);
    break;
  case LEAKYRELU:
    sigmoidLeakyReLU
      (input.sub->features.vec,
       output.sub->features.vec,
       0.01);
    break;
  case VLEAKYRELU:
    sigmoidLeakyReLU
      (input.sub->features.vec,
       output.sub->features.vec,
       0.333);
    break;
  case SOFTMAX:
    sigmoidSoftmax (input.sub->features.vec,output.sub->features.vec,output.featuresPresent.size());
    break;
  case NOSIGMOID:
    break;
  }
}

void applySigmoidBackProp(SpatiallySparseBatchInterface& input, SpatiallySparseBatchInterface& output, ActivationFunction fn) {
  switch(fn) {
  case TANH:
    sigmoidBackpropTanh
      (input.sub->features.vec,output.sub->features.vec,
       input.sub->dfeatures.vec,
       output.sub->dfeatures.vec);
    break;
  case RELU:
    sigmoidBackpropReLU
      (input.sub->features.vec,output.sub->features.vec,
       input.sub->dfeatures.vec,
       output.sub->dfeatures.vec);
    break;
  case LEAKYRELU:
    sigmoidBackpropLeakyReLU
      (input.sub->features.vec,
       output.sub->features.vec,
       input.sub->dfeatures.vec,
       output.sub->dfeatures.vec,
       0.01);
    break;
  case VLEAKYRELU:
    sigmoidBackpropLeakyReLU
      (input.sub->features.vec,
       output.sub->features.vec,
       input.sub->dfeatures.vec,
       output.sub->dfeatures.vec,
       0.333);
    break;
  case SOFTMAX:
    sigmoidBackpropSoftmax
      (input.sub->features.vec,output.sub->features.vec, input.sub->dfeatures.vec,output.sub->dfeatures.vec, output.featuresPresent.size());   break;
  case NOSIGMOID:
    break;
  }
}

SigmoidLayer::SigmoidLayer(ActivationFunction fn) : fn(fn) {
  std::cout << sigmoidNames[fn] << std::endl;
};
void SigmoidLayer::preprocess
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  output.nFeatures=input.nFeatures;
  output.featuresPresent.vec=input.featuresPresent.vec;
  output.spatialSize=input.spatialSize;
  output.nSpatialSites=input.nSpatialSites;
  output.grids=input.grids;
  output.backpropErrors=input.backpropErrors;
}
void SigmoidLayer::forwards(SpatiallySparseBatch &batch,
                            SpatiallySparseBatchInterface &input,
                            SpatiallySparseBatchInterface &output) {
  output.sub->features.resize(output.nSpatialSites*output.featuresPresent.size());
  applySigmoid(input, output, fn);
}
void SigmoidLayer::backwards(SpatiallySparseBatch &batch,
                             SpatiallySparseBatchInterface &input,
                             SpatiallySparseBatchInterface &output,
                             float learningRate,
                             float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
    applySigmoidBackProp(input, output, fn);
  }
}
int SigmoidLayer::calculateInputSpatialSize(int outputSpatialSize) {
  return outputSpatialSize;
}
