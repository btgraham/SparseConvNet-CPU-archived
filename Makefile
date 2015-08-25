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

cifar10: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetCIFAR10.o cifar10.o
	$(CC) -o cifar10 $(OBJ) OpenCVPicture.o SpatiallySparseDatasetCIFAR10.o cifar10.o $(LIBS) $(CFLAGS)

cifar10FullyConnected: $(OBJ) SpatiallySparseDatasetCIFAR10FullyConnected.o cifar10FullyConnected.o
	$(CC) -o cifar10FullyConnected $(OBJ) SpatiallySparseDatasetCIFAR10FullyConnected.o cifar10FullyConnected.o $(LIBS) $(CFLAGS)

cifar100: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetCIFAR100.o cifar100.o
	$(CC) -o cifar100 $(OBJ) OpenCVPicture.o SpatiallySparseDatasetCIFAR100.o cifar100.o $(LIBS) $(CFLAGS)

shrec2015: $(OBJ) Off3DFormatPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015.o
	$(CC) -o shrec2015 $(OBJ) Off3DFormatPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015.o $(LIBS) $(CFLAGS)

shrec2015_: $(OBJ) Off3DFormatPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015_.o
	$(CC) -o shrec2015_ $(OBJ) Off3DFormatPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015_.o $(LIBS) $(CFLAGS)

casia3d: $(OBJ) OnlineHandwritingPicture.o SpatiallySparseDatasetCasiaOLHWDB.o casia3d.o
	$(CC) -o casia3d $(OBJ) OnlineHandwritingPicture.o SpatiallySparseDatasetCasiaOLHWDB.o casia3d.o $(LIBS) $(CFLAGS)

cifar10triangular: $(OBJ) OpenCVTriangularPicture.o SpatiallySparseDatasetCIFAR10.o cifar10triangular.o
	$(CC) -o cifar10triangular $(OBJ) OpenCVTriangularPicture.o SpatiallySparseDatasetCIFAR10.o cifar10triangular.o $(LIBS) $(CFLAGS)

shrec2015triangular: $(OBJ) Off3DFormatTriangularPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015triangular.o
	$(CC) -o shrec2015triangular $(OBJ) Off3DFormatTriangularPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015triangular.o $(LIBS) $(CFLAGS)

cvap_rha: $(OBJ) CVAP_RHA_Picture.o SpatiallySparseDatasetCVAP_RHA.o cvap_rha.o
	$(CC) -o cvap_rha $(OBJ) CVAP_RHA_Picture.o SpatiallySparseDatasetCVAP_RHA.o cvap_rha.o $(LIBS) $(CFLAGS)

ucf101: $(OBJ) UCF101Picture.o SpatiallySparseDatasetUCF101.o ucf101.o
	$(CC) -o ucf101 $(OBJ) UCF101Picture.o SpatiallySparseDatasetUCF101.o ucf101.o $(LIBS) $(CFLAGS)

imagenet2012triangular: $(OBJ) OpenCVTriangularPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012triangular.o
	$(CC) -o imagenet2012triangular $(OBJ) OpenCVTriangularPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012triangular.o $(LIBS) $(CFLAGS)

mnist: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetMnist.o mnist.o
	$(CC) -o mnist $(OBJ) OpenCVPicture.o SpatiallySparseDatasetMnist.o mnist.o $(LIBS) $(CFLAGS)

plankton: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetOpenCV.o plankton.o
	$(CC) -o plankton $(OBJ) OpenCVPicture.o SpatiallySparseDatasetOpenCV.o plankton.o $(LIBS) $(CFLAGS)

plankton_jluo_30july2015: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetOpenCV.o plankton_jluo_30july2015.o
	$(CC) -o plankton_jluo_30july2015 $(OBJ) OpenCVPicture.o SpatiallySparseDatasetOpenCV.o plankton_jluo_30july2015.o $(LIBS) $(CFLAGS)
plankton_jluo_30july2015_2: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetOpenCV.o plankton_jluo_30july2015_2.o
	$(CC) -o plankton_jluo_30july2015_2 $(OBJ) OpenCVPicture.o SpatiallySparseDatasetOpenCV.o plankton_jluo_30july2015_2.o $(LIBS) $(CFLAGS)

cifar10indexLearning: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetCIFAR10.o cifar10indexLearning.o
	$(CC) -o cifar10indexLearning $(OBJ) OpenCVPicture.o SpatiallySparseDatasetCIFAR10.o cifar10indexLearning.o $(LIBS) $(CFLAGS)

imagenet2012: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012.o
	$(CC) -o imagenet2012 $(OBJ) OpenCVPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012.o $(LIBS) $(CFLAGS)
imagenet2012fmp: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012fmp.o
	$(CC) -o imagenet2012fmp $(OBJ) OpenCVPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012fmp.o $(LIBS) $(CFLAGS)
imagenet2012fmp2: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012fmp2.o
	$(CC) -o imagenet2012fmp2 $(OBJ) OpenCVPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012fmp2.o $(LIBS) $(CFLAGS)
imagenet2012fmp3: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012fmp3.o
	$(CC) -o imagenet2012fmp3 $(OBJ) OpenCVPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012fmp3.o $(LIBS) $(CFLAGS)
imagenet2012fmp4: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012fmp4.o
	$(CC) -o imagenet2012fmp4 $(OBJ) OpenCVPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012fmp4.o $(LIBS) $(CFLAGS)
imagenet2012fmp5: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012fmp5.o
	$(CC) -o imagenet2012fmp5 $(OBJ) OpenCVPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012fmp5.o $(LIBS) $(CFLAGS)
imagenet2012fmp5s: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012fmp5s.o
	$(CC) -o imagenet2012fmp5s $(OBJ) OpenCVPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012fmp5s.o $(LIBS) $(CFLAGS)
imagenet2012alexnetTriangular: $(OBJ) OpenCVTriangularPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012alexnetTriangular.o
	$(CC) -o imagenet2012alexnetTriangular $(OBJ) OpenCVTriangularPicture.o SpatiallySparseDatasetImageNet2012.o imagenet2012alexnetTriangular.o $(LIBS) $(CFLAGS)


scenes_sun: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetOpenCV.o scenes_sun.o
	$(CC) -o scenes_sun $(OBJ) OpenCVPicture.o SpatiallySparseDatasetOpenCV.o scenes_sun.o $(LIBS) $(CFLAGS)
scenes_sun2: $(OBJ) OpenCVPicture.o SpatiallySparseDatasetOpenCV.o scenes_sun2.o
	$(CC) -o scenes_sun2 $(OBJ) OpenCVPicture.o SpatiallySparseDatasetOpenCV.o scenes_sun2.o $(LIBS) $(CFLAGS)
