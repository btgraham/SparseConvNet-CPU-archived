//Average everything that makes it to the final layer

#define TERMINAL_POOLING_MAX_ACTIVE_SITES 1024
#include <iostream>
#include <cassert>
#include "utilities.h"
#include "cudaUtilities.h"
#include "TerminalPoolingLayer.h"

void terminalGridPoolingRules
(SparseGrid &inputGrid,
 SparseGrid &outputGrid,
 int S,
 int &nOutputSpatialSites,
 std::vector<int>& rules) {
  assert(inputGrid.mp.size()<=TERMINAL_POOLING_MAX_ACTIVE_SITES); //Upper bound for ease of kernel memory management
  //std::cout << inputGrid.mp.size() << std::endl;
  if (inputGrid.mp.size()==0) { //Danger, total loss of information
    rules.push_back(inputGrid.backgroundCol);
  } else {
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter)
      rules.push_back(iter->second);
  }
  outputGrid.mp[0]=nOutputSpatialSites++;
  rules.resize(S*nOutputSpatialSites,-1); //pad with -1 values
}

void terminalPool(std::vector<float>& g1, std::vector<float>& g2, std::vector<int>& rules, int count, int ps2, int nOut) {
  for (int row=0;row<count;row++) {
    for (int j=0;j<nOut;++j) {
      g2[row*nOut+j]=0;
    }
    int p=0;
    for (;p<ps2;++p) {
      int r=rules[row*ps2+p]*nOut;
      if (r<0) break;
      for (int j=0;j<nOut;++j) {
        g2[row*nOut+j]+=g1[r+j];
      }
    }
    for (int j=0;j<nOut;++j) {
      g2[row*nOut+j]/=p;
    }
  }
}

void terminalPoolBackProp(std::vector<float>& d1, std::vector<float>& d2, std::vector<int>& rules, int count, int nOut, int ps2) {
  for (int row=0;row<count;row++) {
    int maxP=0;
    while (maxP<ps2 and rules[row*ps2+maxP]>=0)
      maxP++;
    int p=0;
    for (;p<maxP;++p) {
      int r=rules[row*ps2+p]*nOut;
      for (int j=0;j<nOut;++j) {
        d1[r+j]=d2[row*nOut+j]/maxP;
      }
    }
  }
}

TerminalPoolingLayer::TerminalPoolingLayer(int poolSize, int S)
  : inSpatialSize(poolSize), outSpatialSize(1), poolSize(poolSize), S(S) {
  std::cout << "TerminalPooling " << poolSize << " " << S << std::endl;
}
void TerminalPoolingLayer::preprocess
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
  for (int item=0;item<batch.batchSize;item++)
    terminalGridPoolingRules
      (input.grids[item],
       output.grids[item],
       S,
       output.nSpatialSites,
       output.rules.vec);
}
void TerminalPoolingLayer::forwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  output.sub->poolingChoices.resize(output.nSpatialSites*output.featuresPresent.size());
  output.sub->features.resize(output.nSpatialSites*output.featuresPresent.size());
  terminalPool(input.sub->features.vec,output.sub->features.vec,output.rules.vec,output.nSpatialSites,S,output.featuresPresent.size());
}
void TerminalPoolingLayer::backwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output,
 float learningRate,
 float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
    input.sub->dfeatures.setZero();
    terminalPoolBackProp
      (input.sub->dfeatures.vec, output.sub->dfeatures.vec, output.rules.vec,output.nSpatialSites, output.featuresPresent.size(),S);
  }
}
int TerminalPoolingLayer::calculateInputSpatialSize(int outputSpatialSize) {
  assert(outputSpatialSize==1);
  std::cout << "(" << outSpatialSize <<"TP" <<inSpatialSize << ") ";
  return inSpatialSize;
}
