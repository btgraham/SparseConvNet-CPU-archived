#include <iostream>
#include <cassert>
#include "utilities.h"
#include "cudaUtilities.h"
#include "MaxPoolingLayer.h"
#include "Regions.h"

void maxPool(std::vector<float>& g1, std::vector<float>& g2, std::vector<int>& rules, int count, int sd, int nOut, std::vector<int>& d_choice) {
  for (int row=0;row<count;row++) {
    for (int s=0;s<sd;s++) {
      int r=rules[row*sd+s];
      if (r>=0) {
        for (int i=0;i<nOut;i++) {
          if (s==0 or g2[row*nOut+i]<g1[r*nOut+i]) {
            g2[row*nOut+i]=g1[r*nOut+i];
            d_choice[row*nOut+i]=r*nOut+i;
          }
        }
      }
    }
  }
}

void maxPoolBackProp(std::vector<float>& d1, std::vector<float>& d2, int count, int nOut, std::vector<int>& d_choice) {
  for (int row=0;row<count;row++) {
    for (int i=0;i<nOut;i++) {
      d1[d_choice[row*nOut+i]]+=d2[row*nOut+i];
    }
  }
}

//TODO: Refactor the different pooling classes somehow


MaxPoolingLayer::MaxPoolingLayer(int poolSize, int poolStride, int dimension)
  : poolSize(poolSize), poolStride(poolStride), dimension(dimension) {
  sd=ipow(poolSize,dimension);
  std::cout << "MaxPooling " << poolSize << " " << poolStride << std::endl;
}
void MaxPoolingLayer::preprocess
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  assert(input.spatialSize==inSpatialSize);
  output.nFeatures=input.nFeatures;
  output.featuresPresent.vec=input.featuresPresent.vec;
  output.spatialSize=outSpatialSize;
  output.nSpatialSites=0;
  output.grids.resize(batch.batchSize);
  output.backpropErrors=input.backpropErrors;
  RegularPoolingRegions regions(inSpatialSize, outSpatialSize,dimension,poolSize, poolStride);
  for (int item=0;item<batch.batchSize;item++)
    gridRules
      (input.grids[item],
       output.grids[item],
       regions,
       output.nSpatialSites,
       output.rules.vec);
}
void MaxPoolingLayer::forwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  output.sub->poolingChoices.resize(output.nSpatialSites*output.featuresPresent.size());
  output.sub->features.resize(output.nSpatialSites*output.featuresPresent.size());
  maxPool(input.sub->features.vec,output.sub->features.vec,output.rules.vec,output.nSpatialSites,sd,output.featuresPresent.size(),output.sub->poolingChoices.vec);
}
void MaxPoolingLayer::backwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output,
 float learningRate,
 float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
    input.sub->dfeatures.setZero();
    maxPoolBackProp
      (input.sub->dfeatures.vec, output.sub->dfeatures.vec, output.nSpatialSites, output.featuresPresent.size(), output.sub->poolingChoices.vec);
  }
}
int MaxPoolingLayer::calculateInputSpatialSize(int outputSpatialSize) {
  outSpatialSize=outputSpatialSize;
  inSpatialSize=poolSize+(outputSpatialSize-1)*poolStride;
  std::cout << "(" << outSpatialSize <<"," <<inSpatialSize << ") ";
  return inSpatialSize;
}

PseudorandomOverlappingFractionalMaxPoolingLayer::PseudorandomOverlappingFractionalMaxPoolingLayer(int poolSize, float fmpShrink, int dimension) : poolSize(poolSize), fmpShrink(fmpShrink), dimension(dimension) {
  sd=ipow(poolSize,dimension);
  std::cout << "Pseudorandom overlapping Fractional Max Pooling " << fmpShrink << " " << poolSize << std::endl;
}
void PseudorandomOverlappingFractionalMaxPoolingLayer::preprocess
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  assert(input.spatialSize==inSpatialSize);
  output.nFeatures=input.nFeatures;
  output.featuresPresent.vec=input.featuresPresent.vec;
  output.spatialSize=outSpatialSize;
  output.nSpatialSites=0;
  output.grids.resize(batch.batchSize);
  output.backpropErrors=input.backpropErrors;
  PseudorandomOverlappingFractionalPoolingRegions regions(inSpatialSize, outSpatialSize,dimension, poolSize,rng);
  for (int item=0;item<batch.batchSize;item++) {
    gridRules
      (input.grids[item],
       output.grids[item],
       regions,
       output.nSpatialSites,
       output.rules.vec);
  }
}
void PseudorandomOverlappingFractionalMaxPoolingLayer::forwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  output.sub->poolingChoices.resize(output.nSpatialSites*output.featuresPresent.size());
  output.sub->features.resize(output.nSpatialSites*output.featuresPresent.size());
  maxPool(input.sub->features.vec,output.sub->features.vec,output.rules.vec,output.nSpatialSites,sd,output.featuresPresent.size(),output.sub->poolingChoices.vec);
}
void PseudorandomOverlappingFractionalMaxPoolingLayer::backwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output,
 float learningRate,
 float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
    input.sub->dfeatures.setZero();
    maxPoolBackProp
      (input.sub->dfeatures.vec, output.sub->dfeatures.vec, output.nSpatialSites, output.featuresPresent.size(), output.sub->poolingChoices.vec);
  }
}
int PseudorandomOverlappingFractionalMaxPoolingLayer::calculateInputSpatialSize
(int outputSpatialSize) {
  outSpatialSize=outputSpatialSize;
  inSpatialSize=outputSpatialSize*fmpShrink+0.5;
  if (inSpatialSize==outputSpatialSize)
    inSpatialSize++;
  std::cout << "(" << outSpatialSize <<"," <<inSpatialSize << ") ";
  return inSpatialSize;
}

RandomOverlappingFractionalMaxPoolingLayer::RandomOverlappingFractionalMaxPoolingLayer
(int poolSize, float fmpShrink, int dimension)
  : poolSize(poolSize), fmpShrink(fmpShrink), dimension(dimension) {
  sd=ipow(poolSize,dimension);
  std::cout << "Random overlapping Fractional Max Pooling " << fmpShrink << " " << poolSize << std::endl;
}
void RandomOverlappingFractionalMaxPoolingLayer::preprocess
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  assert(input.spatialSize==inSpatialSize);
  output.nFeatures=input.nFeatures;
  output.featuresPresent.vec=input.featuresPresent.vec;
  output.spatialSize=outSpatialSize;
  output.nSpatialSites=0;
  output.grids.resize(batch.batchSize);
  output.backpropErrors=input.backpropErrors;
  RandomOverlappingFractionalPoolingRegions regions(inSpatialSize, outSpatialSize,dimension, poolSize,rng);
  for (int item=0;item<batch.batchSize;item++)
    gridRules
      (input.grids[item],
       output.grids[item],
       regions,
       output.nSpatialSites,
       output.rules.vec);
}
void RandomOverlappingFractionalMaxPoolingLayer::forwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  output.sub->poolingChoices.resize(output.nSpatialSites*output.featuresPresent.size());
  output.sub->features.resize(output.nSpatialSites*output.featuresPresent.size());
  maxPool(input.sub->features.vec,output.sub->features.vec,output.rules.vec,output.nSpatialSites,sd,output.featuresPresent.size(),output.sub->poolingChoices.vec);
}
void RandomOverlappingFractionalMaxPoolingLayer::backwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output,
 float learningRate,
 float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
    input.sub->dfeatures.setZero();
    maxPoolBackProp
      (input.sub->dfeatures.vec, output.sub->dfeatures.vec, output.nSpatialSites, output.featuresPresent.size(), output.sub->poolingChoices.vec);
  }
}
int RandomOverlappingFractionalMaxPoolingLayer::calculateInputSpatialSize
(int outputSpatialSize) {
  outSpatialSize=outputSpatialSize;
  inSpatialSize=outputSpatialSize*fmpShrink+0.5;
  if (inSpatialSize==outputSpatialSize)
    inSpatialSize++;
  std::cout << "(" << outSpatialSize <<"," <<inSpatialSize << ") ";
  return inSpatialSize;
}
