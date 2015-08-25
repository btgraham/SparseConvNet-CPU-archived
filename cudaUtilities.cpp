#define CUDAUTILITIES_CU
#include "cudaUtilities.h"
#include <iostream>
#include <cassert>
#include <cblas.h>
#include <cmath>

int intRoundUp(int a, int d) {
  return ((a+d-1)/d)*d;
}
int intRound(int a, int d) {
  return round(a*1.0/d)*d;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//GEMM for matrices in row major form. ///////////////////////////////////////////////////////////
//A is l*m, B is m*r, C is l*r. Set C to alpha A B + beta C.
void d_rowMajorSGEMM_alphaAB_betaC (std::vector<float>& A, std::vector<float>& B, std::vector<float>& C,
                                    int l, int m, int r,
                                    float alpha, float beta)
{
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,l,r,m,alpha,&A[0],m,&B[0],r,beta,&C[0],r);
}
//A^t is l*m, B is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtB_betaC (std::vector<float>& A, std::vector<float>& B, std::vector<float>& C,
                                     int l, int m, int r,
                                     float alpha, float beta)
{
  cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,l,r,m,alpha,&A[0],l,&B[0],r,beta,&C[0],r);
}
//A is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaABt_betaC (std::vector<float>& A, std::vector<float>& B, std::vector<float>& C,
                                     int l, int m, int r,
                                     float alpha, float beta)
{
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,l,r,m,alpha,&A[0],m,&B[0],m,beta,&C[0],r);
}
//A^t is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtBt_betaC (std::vector<float>& A, std::vector<float>& B, std::vector<float>& C,
                                      int l, int m, int r,
                                      float alpha, float beta)
{
  cblas_sgemm(CblasRowMajor,CblasTrans,CblasTrans,l,r,m,alpha,&A[0],l,&B[0],m,beta,&C[0],r);
}
