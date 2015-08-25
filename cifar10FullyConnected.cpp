#include "SparseConvNet.h"
#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetCIFAR10FullyConnected.h"
#include <iostream>

int epoch=0;
int cudaDevice=-1; //PCI bus ID, -1 for default GPU
int batchSize=100;

class ANN : public SparseConvNet {
public:
  ANN (ActivationFunction fn, int nInputFeatures, int nClasses, float p=0.0f, int cudaDevice=-1, int nTop=1);
};
ANN::ANN
(ActivationFunction fn,
 int nInputFeatures, int nClasses, float p, int cudaDevice, int nTop)
  : SparseConvNet(1,nInputFeatures, nClasses, cudaDevice, nTop) {
  // addLeNetLayerMP(1024*12,1,1,1,1,fn,p);
  // addLeNetLayerMP(1024*12,1,1,1,1,fn,p);  //1024*12, 2 layers, RELU, test errors 37.87%
  addLeNetLayerMP(256,1,1,1,1,NOSIGMOID);
  addLeNetLayerMP(1024,1,1,1,1,fn,p);
  addLeNetLayerMP(256,1,1,1,1,NOSIGMOID);
  addLeNetLayerMP(1024,1,1,1,1,fn,p);
  addSoftmaxLayer();
}

int main() {
  std::string baseName="weights/cifar10";

  SpatiallySparseDataset trainSet=Cifar10TrainSetFullyConnected();
  SpatiallySparseDataset testSet=Cifar10TestSetFullyConnected();

  trainSet.summary();
  testSet.summary();
  ANN ann(RELU,trainSet.nFeatures,trainSet.nClasses,0.5f,cudaDevice);

  if(epoch>0)
    ann.loadWeights(baseName,epoch);
  for (epoch++;;epoch++) {
    std::cout <<"epoch: " << epoch << " " << std::flush;
    ann.processDataset(trainSet, batchSize,0.003*exp(-0.005 * epoch),0.99);
    if (epoch%5==0) {
      ann.saveWeights(baseName,epoch);
      ann.processDataset(testSet,  batchSize);
    }
  }
}
