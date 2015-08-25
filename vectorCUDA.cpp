#include "vectorCUDA.h"
#include <iostream>
#include "Rng.h"
#include "cudaUtilities.h"
#include <cstring>

template <typename t> int vectorCUDA<t>::size() {
  return vec.size();
}
template <typename t> float vectorCUDA<t>::meanAbs(float negWeight) {
  float total=0;
  for (int i=0;i<size();i++)
    total+=(vec[i]>0)?vec[i]:(-negWeight*vec[i]);
  if (total!=total) {
    std::cout << "NaN in vectorCUDA<t>::meanAbs()\n";
    exit(1);
  }
  return total/size();
}
template <typename t> void vectorCUDA<t>::multiplicativeRescale(float multiplier) {
  for (int i=0;i<size();i++)
    vec[i]*=multiplier;
}
template <typename t> void vectorCUDA<t>::setZero() {
  memset(&vec[0],0,sizeof(t)*vec.size());
}
template <typename t> void vectorCUDA<t>::setConstant(float a) {
  for (int i=0;i<vec.size();i++)
    vec[i]=a;
}
template <typename t> void vectorCUDA<t>::setUniform(float a,float b) {
  RNG rng;
  for (int i=0;i<vec.size();i++)
    vec[i]=rng.uniform(a,b);
}
template <typename t> void vectorCUDA<t>::setBernoulli(float p) {
  RNG rng;
  for (int i=0;i<vec.size();i++)
    vec[i]=rng.bernoulli(p);
}
template <typename t> void vectorCUDA<t>::setNormal(float mean, float sd) {
  RNG rng;
  for (int i=0;i<vec.size();i++)
    vec[i]=rng.normal(mean,sd);
}
template <typename t> void vectorCUDA<t>::resize(int n) {
  vec.resize(n);
}
template <typename t> vectorCUDA<t>::vectorCUDA(int dsize) {
  vec.resize(dsize);
}
template <typename t> vectorCUDA<t>::~vectorCUDA() {
}

template class vectorCUDA<float>;
template class vectorCUDA<int>;
