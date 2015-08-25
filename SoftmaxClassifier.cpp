#include "SoftmaxClassifier.h"
#include "cudaUtilities.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "utilities.h"

void dDerivativeOfCostWRTpreSoftmaxTopLevelWeights
(int batchSize, std::vector<float>& topDelta, std::vector<float>& topGrid, std::vector<int>& labels, int N) {
  for (int k=0;k<batchSize;k++) {
    for(int i=0;i<N;i++) {
      topDelta[k*N+i]=topGrid[k*N+i]-(i==labels[k]);
    }
  }
}

void SoftmaxClassifier(SpatiallySparseBatchInterface& input, SpatiallySparseBatch& batch, int nTop) {
  //Assume no dropout in the output layer! nClasses:=input.nFeatures.
  assert(batch.batchSize==input.nSpatialSites);
  assert(input.nFeatures==input.featuresPresent.size());

  if (batch.type==TRAINBATCH) {//Begin backprop. Top layer: d Cost / d SoftmaxInput
    input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
    dDerivativeOfCostWRTpreSoftmaxTopLevelWeights
      (batch.batchSize, input.sub->dfeatures.vec, input.sub->features.vec,
       batch.labels.vec, input.nFeatures);
  }

  float* probs=&input.sub->features.vec[0];
  for (int i=0;i<batch.batchSize;++i)
    batch.probabilities.push_back(std::vector<float> (probs+i*input.nFeatures,probs+(i+1)*input.nFeatures));
  for (int i=0;i<batch.batchSize;i++)
    batch.predictions.push_back(vectorTopIndices(batch.probabilities[i],nTop));

  if (batch.type!=UNLABELEDBATCH) {
    batch.mistakes+=batch.batchSize;
    for (int i=0;i<batch.batchSize;i++) {
      batch.negativeLogLikelihood-=log(std::max(batch.probabilities[i][batch.labels.vec[i]],(float)1.0e-15));
      for (int j=0;j<nTop;j++) {
        if (batch.predictions[i][j]==batch.labels.vec[i]) {
          batch.mistakes--;
        }
      }
    }
    std::cout << batch.mistakes << " " <<std::flush;
  }
}
