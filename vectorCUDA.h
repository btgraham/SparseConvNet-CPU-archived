#pragma once
#include <vector>
#include "cudaUtilities.h"

template <typename t> class vectorCUDA {
private:
public:
  std::vector<t> vec;
  vectorCUDA(int dsize=0);
  ~vectorCUDA();
  int size();
  float meanAbs(float negWeight=1);
  void multiplicativeRescale(float multiplier);
  void setZero();
  void setConstant(float a=0);
  void setUniform(float a=0,float b=1);
  void setBernoulli(float p=0.5);
  void setNormal(float mean=0, float sd=1);
  void resize(int n);
};
