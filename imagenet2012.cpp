#include "SparseConvNet.h"
#include "SpatiallySparseDatasetImageNet2012.h"
#include <iostream>
#include <string>

int epoch=37; //terminal pooling till 24
int cudaDevice=-1; //PCI bus ID, -1 for default GPU
int batchSize=2;

Picture* OpenCVPicture::distort(RNG& rng, batchType type) {
  OpenCVPicture* pic=new OpenCVPicture(*this);
  float
    c00=1, c01=0,  //2x2 identity matrix---starting point for calculating affine distortion matrix
    c10=0, c11=1;
  if (type==TRAINBATCH) {
    pic->loadData(256+rng.randint(256));
    pic->colorDistortion(rng, 0.1*255, 0.15*255, 0.8, 0.8);
    float r=rng.uniform(-0.1,0.1);
    float alpha=rng.uniform(-0.3,0.3);
    float beta=rng.uniform(-0.2,0.2)+alpha;
    c00=(1+r)*cos(alpha); c01=(1+r)*sin(alpha);
    c10=-(1-r)*sin(beta); c11=(1-r)*cos(beta);
  } else {
    pic->loadData(384);
  }
  if (rng.randint(2)==0) {c00*=-1; c01*=-1;}//Horizontal flip
  pic->affineTransform(c00, c01, c10, c11);
  pic->jiggleFit(rng,1023);
  return pic;
}

class Imagenet : public SparseConvNet {
public:
  Imagenet (int dimension, int nInputFeatures, int nClasses, int cudaDevice=-1, int nTop=1);
};
Imagenet::Imagenet
(int dimension, int nInputFeatures, int nClasses, int cudaDevice, int nTop)
  : SparseConvNet(dimension,nInputFeatures, nClasses, cudaDevice, nTop) {
  addLeNetLayerMP(  32,3,1,3,2,VLEAKYRELU,0,9);
  addLeNetLayerMP(  64,2,1,3,2,VLEAKYRELU,0,2);
  addLeNetLayerMP( 192,2,1,3,2,VLEAKYRELU,0,2);
  addLeNetLayerMP( 480,2,1,3,2,VLEAKYRELU,0,2);
  addLeNetLayerMP( 832,2,1,3,2,LEAKYRELU);
  addLeNetLayerMP(1024,2,1,3,2,LEAKYRELU);
  addLeNetLayerMP(1024,2,1,3,2,LEAKYRELU);
  addLeNetLayerMP(1024,2,1,3,2,LEAKYRELU);
  addLeNetLayerMP(1024,2,1,1,1,LEAKYRELU);
  addLeNetLayerMP(1024,1,1,1,1,LEAKYRELU,0.4);
  addSoftmaxLayer();

  // addLeNetLayerMP(  32,3,1,3,2,VLEAKYRELU,0,9);
  // addLeNetLayerMP(  64,2,1,3,2,VLEAKYRELU,0,2);
  // addLeNetLayerMP( 192,2,1,3,2,VLEAKYRELU,0,2);
  // addLeNetLayerMP( 480,2,1,3,2,VLEAKYRELU,0,2);
  // addLeNetLayerMP( 832,2,1,3,2,LEAKYRELU,0,2);
  // addLeNetLayerMP(1024,2,1,1,1,LEAKYRELU,0,2);
  // addTerminalPoolingLayer(32);
  // addLeNetLayerMP(1024,1,1,1,1,LEAKYRELU,0.4);
  // addSoftmaxLayer();

}


int main() {
  std::string baseName="weights/imagenet2012";

  auto trainSet=ImageNet2012TrainSet();
  auto validationSet=ImageNet2012ValidationSet();
  trainSet.summary();
  validationSet.summary();

  Imagenet cnn(2,trainSet.nFeatures,trainSet.nClasses,cudaDevice,5);

  if (epoch>0) {
    cnn.loadWeights(baseName,epoch);
    //cnn.processDatasetRepeatTest(validationSubset, batchSize,12);
  }

  for (int i=0;i<100;++i) {
    SpatiallySparseDataset trainSubset=trainSet.subset(320);
    cnn.processDataset(trainSubset, batchSize,0.001,0.999);
  }
  for (epoch++;;epoch++) {
    std::cout <<"epoch: " << epoch << std::endl;
    for (int i=0;i<10;++i) {
      SpatiallySparseDataset trainSubset=trainSet.subset(16000);
      cnn.processDataset(trainSubset, batchSize,0.001,0.999);
      cnn.saveWeights(baseName,epoch);
    }
    auto validationSubset=validationSet.subset(5000);
    cnn.processDatasetRepeatTest(validationSubset, batchSize,3);
  }
}
