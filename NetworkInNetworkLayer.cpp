#include "NetworkInNetworkLayer.h"
#include "cudaUtilities.h"
#include "SigmoidLayer.h"
#include <iostream>
#include <cassert>

void dShrinkMatrixForDropout
(std::vector<float>& m, std::vector<float>& md,
 std::vector<int>& inFeaturesPresent, std::vector<int>& outFeaturesPresent,
 int nOut) {
  for (int i_=0;i_<inFeaturesPresent.size();i_++) {
    int i=i_*outFeaturesPresent.size();
    int ii=inFeaturesPresent[i_]*nOut;
    for (int j=0;j<outFeaturesPresent.size();j++) {
      int jj=outFeaturesPresent[j];
      md[i+j]=m[ii+jj];
    }
  }
}

void dShrinkVectorForDropout(std::vector<float>& m, std::vector<float>& md, std::vector<int>& outFeaturesPresent) {
  for(int i=0; i<outFeaturesPresent.size(); i++) {
    int ii=outFeaturesPresent[i];
    md[i]=m[ii];
  }
}

//Nesterov
void dGradientDescent
(std::vector<float>& d_delta, std::vector<float>& d_momentum, std::vector<float>& d_weights,
 int nIn, int nOut, float learningRate, float momentum) {
  for (int i=0;i<nIn;i++) {
    for(int j=i; j<i+nOut; j++) {
      d_weights[j]-=d_momentum[j]*momentum;
      d_momentum[j]=momentum*d_momentum[j]-learningRate*(1-momentum)*d_delta[j];
      d_weights[j]=d_weights[j]+d_momentum[j]*(1+momentum);
    }
  }
}

void dGradientDescentShrunkMatrix
(std::vector<float>& d_delta, std::vector<float>& d_momentum, std::vector<float>& d_weights,
 int nOut,
 std::vector<int>& inFeaturesPresent, std::vector<int>& outFeaturesPresent,
 float learningRate,
 float momentum) {
  for (int i_=0;i_<inFeaturesPresent.size();i_++) {
    int i=i_*outFeaturesPresent.size();
    int ii=inFeaturesPresent[i_]*nOut;
    for(int j=0; j<outFeaturesPresent.size(); j++) {
      int jj=outFeaturesPresent[j];
      //NAG light
      d_weights[ii+jj]-=d_momentum[ii+jj]*momentum;
      d_momentum[ii+jj]=momentum*d_momentum[ii+jj]-learningRate*(1-momentum)*d_delta[i+j];
      d_weights[ii+jj]=d_weights[ii+jj]+d_momentum[ii+jj]*(1+momentum);
    }
  }
}

void dGradientDescentShrunkVector
(std::vector<float>& d_delta, std::vector<float>& d_momentum, std::vector<float>& d_weights,
 int nOut,
 std::vector<int>& outFeaturesPresent,
 float learningRate,
 float momentum) {
  for(int i=0; i<outFeaturesPresent.size(); i++) {
    int ii=outFeaturesPresent[i];
    //NAG light
    d_weights[ii]-=d_momentum[ii]*momentum;
    d_momentum[ii]=momentum*d_momentum[ii]-learningRate*(1-momentum)*d_delta[i];
    d_weights[ii]=d_weights[ii]+d_momentum[ii]*(1+momentum);
  }
}


void columnSum(std::vector<float>& matrix, std::vector<float>& target, int nRows, int nColumns) {
  for (int row=0;row<nRows;row++) {
    for (int column=0;column<nColumns;column++) {
      target[column]+=matrix[row*nColumns+column];
    }
  }
}

void replicateArray(std::vector<float>& src, std::vector<float>& dst, int nRows, int nColumns) {
  for (int row=0;row<nRows;row++) {
    for (int column=0;column<nColumns;column++) {
      dst[row*nColumns+column]=src[column];
    }
  }
}

NetworkInNetworkLayer::NetworkInNetworkLayer(int nFeaturesIn, int nFeaturesOut,
                                             float dropout,ActivationFunction fn,
                                             float alpha//used to determine intialization weights only
                                             ) :
  nFeaturesIn(nFeaturesIn), nFeaturesOut(nFeaturesOut),
  dropout(dropout), fn(fn),
  W(nFeaturesIn*nFeaturesOut), MW(nFeaturesIn*nFeaturesOut),
  B(nFeaturesOut), MB(nFeaturesOut) {
  float scale=pow(6.0f/(nFeaturesIn+nFeaturesOut*alpha),0.5f);
  W.setUniform(-scale,scale);
  MW.setZero();
  B.setZero();
  MB.setZero();
  std::cout << "Learn " << nFeaturesIn << "->" << nFeaturesOut << " dropout=" << dropout << " " << sigmoidNames[fn] << std::endl;
}
void NetworkInNetworkLayer::preprocess
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  assert(input.nFeatures==nFeaturesIn);
  output.nFeatures=nFeaturesOut;
  output.spatialSize=input.spatialSize;
  output.nSpatialSites=input.nSpatialSites;
  output.grids=input.grids;
  int o=nFeaturesOut*(batch.type==TRAINBATCH?(1.0f-dropout):1.0f);
  output.featuresPresent.vec=rng.NchooseM(nFeaturesOut,o);
  output.backpropErrors=true;
}
void NetworkInNetworkLayer::forwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  output.sub->features.resize(output.nSpatialSites*output.featuresPresent.size());
  if (batch.type==TRAINBATCH and
      nFeaturesIn+nFeaturesOut>input.featuresPresent.size()+output.featuresPresent.size()) {
    w.resize(input.featuresPresent.size()*output.featuresPresent.size());
    dShrinkMatrixForDropout
      (W.vec, w.vec,
       input.featuresPresent.vec,
       output.featuresPresent.vec,
       output.nFeatures);
    b.resize(output.featuresPresent.size());
    dShrinkVectorForDropout(B.vec, b.vec,
                            output.featuresPresent.vec);
    replicateArray(b.vec, output.sub->features.vec, output.nSpatialSites, output.featuresPresent.size());
    d_rowMajorSGEMM_alphaAB_betaC(input.sub->features.vec, w.vec, output.sub->features.vec,
                                  output.nSpatialSites, input.featuresPresent.size(), output.featuresPresent.size(),
                                  1.0f, 1.0f);

  } else {
    replicateArray(B.vec, output.sub->features.vec, output.nSpatialSites, output.featuresPresent.size());
    d_rowMajorSGEMM_alphaAB_betaC(input.sub->features.vec, W.vec, output.sub->features.vec,
                                  output.nSpatialSites, input.nFeatures, output.nFeatures,
                                  1.0f-dropout, 1.0f-dropout);
  }
  multiplyAddCount+=(__int128_t)output.nSpatialSites*input.featuresPresent.size()*output.featuresPresent.size();
  applySigmoid(output, output, fn);
}
void NetworkInNetworkLayer::scaleWeights
(SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output,
 float& scalingUnderneath,
 bool topLayer) {
  assert(input.sub->features.size()>0);
  assert(output.sub->features.size()>0); //call after forwards(...)
  float scale=output.sub->features.meanAbs( (fn==VLEAKYRELU) ? 3 : 100 );
  std::cout << "featureScale:" << scale << std::endl;
  if (topLayer) {
    scale=1;
  } else {
    scale=powf(scale,-0.1); //0.7978846 = sqrt(2/pi) = mean of the half normal distribution
  }
  W.multiplicativeRescale(scale/scalingUnderneath);
  B.multiplicativeRescale(scale);
  MW.multiplicativeRescale(scale/scalingUnderneath);
  MB.multiplicativeRescale(scale);
  scalingUnderneath=scale;
}
void NetworkInNetworkLayer::backwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output,
 float learningRate,
 float momentum) {
  applySigmoidBackProp(output, output, fn);
  dw.resize(input.featuresPresent.size()*output.featuresPresent.size());
  db.resize(output.featuresPresent.size());
  d_rowMajorSGEMM_alphaAtB_betaC(input.sub->features.vec, output.sub->dfeatures.vec, dw.vec,
                                 input.featuresPresent.size(), output.nSpatialSites, output.featuresPresent.size(),
                                 1.0, 0.0);

  multiplyAddCount+=(__int128_t)output.nSpatialSites*input.featuresPresent.size()*output.featuresPresent.size();
  db.setZero();
  columnSum(output.sub->dfeatures.vec, db.vec, output.nSpatialSites, output.featuresPresent.size());

  if (nFeaturesIn+nFeaturesOut>input.featuresPresent.size()+output.featuresPresent.size()) {
    if (input.backpropErrors) {
      input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
      d_rowMajorSGEMM_alphaABt_betaC(output.sub->dfeatures.vec, w.vec, input.sub->dfeatures.vec,
                                     output.nSpatialSites,output.featuresPresent.size(),input.featuresPresent.size(),
                                     1.0, 0.0);
      multiplyAddCount+=(__int128_t)output.nSpatialSites*input.featuresPresent.size()*output.featuresPresent.size();
    }

    dGradientDescentShrunkMatrix
      (dw.vec, MW.vec, W.vec,
       output.nFeatures,
       input.featuresPresent.vec, output.featuresPresent.vec,
       learningRate,momentum);

    dGradientDescentShrunkVector
      (db.vec, MB.vec, B.vec,
       output.nFeatures,
       output.featuresPresent.vec,
       learningRate,momentum);
  } else {
    if (input.backpropErrors) {
      input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
      d_rowMajorSGEMM_alphaABt_betaC(output.sub->dfeatures.vec, W.vec, input.sub->dfeatures.vec,
                                     output.nSpatialSites,nFeaturesOut,nFeaturesIn,
                                     1.0, 0.0);
      multiplyAddCount+=(__int128_t)output.nSpatialSites*input.featuresPresent.size()*output.featuresPresent.size();
    }
    dGradientDescent
      (dw.vec, MW.vec, W.vec, nFeaturesIn, nFeaturesOut, learningRate,momentum);
    dGradientDescent
      (db.vec, MB.vec, B.vec, 1,           nFeaturesOut, learningRate,momentum);
  }
}
void NetworkInNetworkLayer::loadWeightsFromStream(std::ifstream &f) {
  f.read((char*)&W.vec[0],sizeof(float)*W.size());
  f.read((char*)&B.vec[0],sizeof(float)*B.size());
  MW.setZero();
  MB.setZero();
};
void NetworkInNetworkLayer::putWeightsToStream(std::ofstream &f)  {
  f.write((char*)&W.vec[0],sizeof(float)*W.size());
  f.write((char*)&B.vec[0],sizeof(float)*B.size());
};
int NetworkInNetworkLayer::calculateInputSpatialSize(int outputSpatialSize) {
  return outputSpatialSize;
}
