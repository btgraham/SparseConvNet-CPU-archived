#pragma once
#include <vector>
#include <iostream>

#define NTHREADS 512
#define KERNELBLOCKSIZE 32

int intRound(int a, int d);
int intRoundUp(int a, int d);

//////////////////////////////////////////////////////////////////////////////////////////////////
//GEMM for matrices in row major form. ///////////////////////////////////////////////////////////
//A is l*m, B is m*r, C is l*r. Set C to alpha A B + beta C.
void d_rowMajorSGEMM_alphaAB_betaC (std::vector<float>& A, std::vector<float>& B, std::vector<float>& C,
                                    int l, int m, int r,
                                    float alpha, float beta);
//A^t is l*m, B is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtB_betaC (std::vector<float>& A, std::vector<float>& B, std::vector<float>& C,
                                     int l, int m, int r,
                                     float alpha, float beta);
//A is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaABt_betaC (std::vector<float>& A, std::vector<float>& B, std::vector<float>& C,
                                     int l, int m, int r,
                                     float alpha, float beta);
//A^t is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtBt_betaC (std::vector<float>& A, std::vector<float>& B, std::vector<float>& C,
                                      int l, int m, int r,
                                      float alpha, float beta);
///////////////////////////////////////////////////////////////////////////////////////////////////
