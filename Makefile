CC=g++
CFLAGS=--std=c++11 -O3
OBJ=BatchProducer.o ConvolutionalLayer.o ConvolutionalTriangularLayer.o IndexLearnerLayer.o MaxPoolingLayer.o MaxPoolingTriangularLayer.o NetworkArchitectures.o NetworkInNetworkLayer.o Picture.o Regions.o Rng.o SigmoidLayer.o SoftmaxClassifier.o SparseConvNet.o SparseConvNetCUDA.o SpatiallySparseBatch.o SpatiallySparseBatchInterface.o SpatiallySparseDataset.o SpatiallySparseLayer.o TerminalPoolingLayer.o cudaUtilities.o readImageToMat.o types.o utilities.o vectorCUDA.o vectorHash.o
LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc -lrt -larmadillo -lopenblas

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm *.o
casia: $(OBJ) OnlineHandwritingPicture.o SpatiallySparseDatasetCasiaOLHWDB.o casia.o
	$(CC) -o casia $(OBJ) OnlineHandwritingPicture.o SpatiallySparseDatasetCasiaOLHWDB.o casia.o $(LIBS) $(CFLAGS)

casia3d: $(OBJ) OnlineHandwritingPicture.o SpatiallySparseDatasetCasiaOLHWDB.o casia3d.o
	$(CC) -o casia3d $(OBJ) OnlineHandwritingPicture.o SpatiallySparseDatasetCasiaOLHWDB.o casia3d.o $(LIBS) $(CFLAGS)

cifar10: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetCIFAR10.o cifar10.o
	$(CC) -o cifar10 $(OBJ) OpenCVPicture.o SpatiallySparseDatasetCIFAR10.o cifar10.o $(LIBS) $(CFLAGS)

cifar10fmp: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetCIFAR10.o cifar10fmp.o
	$(CC) -o cifar10fmp $(OBJ) OpenCVPicture.o SpatiallySparseDatasetCIFAR10.o cifar10fmp.o $(LIBS) $(CFLAGS)

cifar10FullyConnected: $(OBJ) SpatiallySparseDatasetCIFAR10FullyConnected.o cifar10FullyConnected.o
	$(CC) -o cifar10FullyConnected $(OBJ) SpatiallySparseDatasetCIFAR10FullyConnected.o cifar10FullyConnected.o $(LIBS) $(CFLAGS)

cifar10indexLearning: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetCIFAR10.o cifar10indexLearning.o
	$(CC) -o cifar10indexLearning $(OBJ) OpenCVPicture.o SpatiallySparseDatasetCIFAR10.o cifar10indexLearning.o $(LIBS) $(CFLAGS)

cifar100: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetCIFAR100.o cifar100.o
	$(CC) -o cifar100 $(OBJ) OpenCVPicture.o SpatiallySparseDatasetCIFAR100.o cifar100.o $(LIBS) $(CFLAGS)

cifar10triangular: $(OBJ) OpenCVTriangularPicture.o SpatiallySparseDatasetCIFAR10.o cifar10triangular.o
	$(CC) -o cifar10triangular $(OBJ) OpenCVTriangularPicture.o SpatiallySparseDatasetCIFAR10.o cifar10triangular.o $(LIBS) $(CFLAGS)

shrec2015triangular: $(OBJ) Off3DFormatTriangularPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015triangular.o
	$(CC) -o shrec2015triangular $(OBJ) Off3DFormatTriangularPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015triangular.o $(LIBS) $(CFLAGS)

cvap_rha: $(OBJ) CVAP_RHA_Picture.o SpatiallySparseDatasetCVAP_RHA.o cvap_rha.o
	$(CC) -o cvap_rha $(OBJ) CVAP_RHA_Picture.o SpatiallySparseDatasetCVAP_RHA.o cvap_rha.o $(LIBS) $(CFLAGS)

mnist: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetMnist.o mnist.o
	$(CC) -o mnist $(OBJ) OpenCVPicture.o SpatiallySparseDatasetMnist.o mnist.o $(LIBS) $(CFLAGS)

imagenet2012: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012.o
	$(CC) -o imagenet2012 $(OBJ) OpenCVPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012.o $(LIBS) $(CFLAGS)
