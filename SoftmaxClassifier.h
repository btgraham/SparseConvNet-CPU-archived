#pragma once
#include "SpatiallySparseBatchInterface.h"
#include "SpatiallySparseBatch.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//Calculate softmaxProbability(i) - indicator(i=label)
// for i=0,1,...N-1 with N the number of character classes.
void dDerivativeOfCostWRTpreSoftmaxTopLevelWeights
(int batchSize, std::vector<float>& topDelta, std::vector<float>& topGrid, std::vector<int>& labels, int N);
void SoftmaxClassifier(SpatiallySparseBatchInterface& input, SpatiallySparseBatch& batch, int nTop);
