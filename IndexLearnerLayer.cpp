#include "IndexLearnerLayer.h"
#include "cudaUtilities.h"
#include "SigmoidLayer.h"
#include <iostream>
#include <cassert>
#include "SoftmaxClassifier.h"
#include "NetworkInNetworkLayer.h"
#include "utilities.h"

void dGradientDescentShrunkMatrixNoMomentum
(std::vector<float>& d_delta, std::vector<float>& d_weights,
 int nOut, int nInDropout, int nOutDropout,
 std::vector<int>& inFeaturesPresent, std::vector<int>& outFeaturesPresent,
 float learningRate) {
  for (int i=0;i<nInDropout;i++) {
    int ii=inFeaturesPresent[i]*nOut;
    for(int j=0; j<nOutDropout; j++) {
      int jj=outFeaturesPresent[j];
      //no momentum, weight updated infrequently if the dataset is much larger than each minibatch
      d_weights[ii+jj]-=learningRate*d_delta[i+j];
    }
  }
}

IndexLearnerLayer::IndexLearnerLayer(int nFeaturesIn, int nFeaturesOut) :
  nFeaturesIn(nFeaturesIn), nFeaturesOut(nFeaturesOut) {
  std::cout << "IndexLearnerLayer" << std::endl;
  float scale=pow(6.0f/(nFeaturesIn+nFeaturesOut),0.5f);
  W.resize (nFeaturesIn*nFeaturesOut); W.setZero();//Uniform(-scale,scale);
  MW.resize (nFeaturesIn*nFeaturesOut); MW.setZero();
}
void IndexLearnerLayer::preprocess
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  if (batch.type==TRAINBATCH); {
    assert(input.nFeatures==nFeaturesIn);
    output.nFeatures=nFeaturesOut;
    output.spatialSize=input.spatialSize;
    output.nSpatialSites=input.nSpatialSites;
    output.grids=input.grids;
    output.backpropErrors=true;
  }
}
void IndexLearnerLayer::forwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  output.featuresPresent.vec=indexLearnerIndices;
  output.sub->features.resize(output.nSpatialSites*output.featuresPresent.size());
  output.sub->dfeatures.resize(output.nSpatialSites*output.featuresPresent.size());
  w.resize(input.featuresPresent.size()*output.featuresPresent.size());
  dShrinkMatrixForDropout(W.vec, w.vec,
                          input.featuresPresent.vec,
                          output.featuresPresent.vec,
                          output.nFeatures,
                          input.featuresPresent.size(),
                          output.featuresPresent.size());
  d_rowMajorSGEMM_alphaAB_betaC(input.sub->features.vec, w.vec, output.sub->features.vec,
                                output.nSpatialSites, input.featuresPresent.size(), output.featuresPresent.size(),
                                1.0f, 0.0f);
  applySigmoid(output, output, SOFTMAX);
}
void IndexLearnerLayer::backwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output,
 float learningRate,
 float momentum) {
  applySigmoidBackProp(output, output, SOFTMAX);
  input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
  dw.resize(input.featuresPresent.size()*output.featuresPresent.size());
  d_rowMajorSGEMM_alphaAtB_betaC(input.sub->features.vec, output.sub->dfeatures.vec, dw.vec,
                                 input.featuresPresent.size(), output.nSpatialSites, output.featuresPresent.size(),
                                 1.0, 0.0);

  if (input.backpropErrors) {
    d_rowMajorSGEMM_alphaABt_betaC(output.sub->dfeatures.vec, w.vec, input.sub->dfeatures.vec,
                                   output.nSpatialSites,output.featuresPresent.size(),input.featuresPresent.size(),
                                   1.0, 0.0);
  }
  dGradientDescentShrunkMatrixNoMomentum
    (dw.vec, W.vec,
     output.nFeatures, input.featuresPresent.size(), output.featuresPresent.size(),
     input.featuresPresent.vec, output.featuresPresent.vec,
     learningRate);
}
void IndexLearnerLayer::loadWeightsFromStream(std::ifstream &f) {
  f.read((char*)&W.vec[0],sizeof(float)*W.size());
};
void IndexLearnerLayer::putWeightsToStream(std::ofstream &f)  {
  f.write((char*)&W.vec[0],sizeof(float)*W.size());
};
int IndexLearnerLayer::calculateInputSpatialSize(int outputSpatialSize) {
  return outputSpatialSize;
}


void IndexLearner(SpatiallySparseBatchInterface& input, SpatiallySparseBatch& batch, int nTop) {
  assert(batch.batchSize==input.nSpatialSites);
  assert(ipow(batch.batchSize,2)==input.sub->features.size());
  assert(batch.type==TRAINBATCH);

  float* probs=&input.sub->features.vec[0];
  for (int i=0;i<batch.batchSize;++i)
    batch.probabilities.push_back(std::vector<float> (probs+i*batch.batchSize,probs+(i+1)*batch.batchSize));
  for (int i=0;i<batch.batchSize;i++)
    batch.predictions.push_back(vectorTopIndices(batch.probabilities[i],nTop));

  batch.mistakes+=batch.batchSize;
  for (int i=0;i<batch.batchSize;i++) {
    batch.negativeLogLikelihood-=log(std::max(batch.probabilities[i][i],(float)1.0e-15));
    for (int j=0;j<nTop;j++) {
      if (batch.predictions[i][j]==i) {
        batch.mistakes--;
      }
    }
  }
  //Begin backprop. Top layer: d Cost / d SoftmaxInput
  vectorCUDA<int> labels;
  labels.vec=range(batch.batchSize);
  input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
  dDerivativeOfCostWRTpreSoftmaxTopLevelWeights
    (batch.batchSize, input.sub->dfeatures.vec, input.sub->features.vec,
     labels.vec, batch.batchSize);
}
