/*
 * Parameter.h
 *
 *  Created on: Dec 6, 2015
 *      Author: chihyang
 */

#ifndef SRC_Parameter_H_
#define SRC_Parameter_H_
#include "tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu, int N>
class ParameterBase{
public:
    Tensor<xpu, N, dtype> weight; // value for the parameter
    Tensor<xpu, N, dtype> grad; // gradients for the parameter
    Tensor<xpu, N, dtype> egrad; // sum gradients for the adagrad

    ParameterBase(){}

    inline void random(dtype bound, int seed)
    {
        random(weight, -1.0*bound, 1.0*bound, seed);    
    }
    void release()
    {
        FreeSpace(&weight);
        FreeSpace(&grad);
        FreeSpace(&egrad);
    }
    inline int size(int dim) const
    {
        return weight.size(dim);
    }
    inline int size(int dim)
    {
        return weight.size(dim);
    }
    inline dtype squarenorm()
    {
        return squarenorm(grad);
    }
    inline dtype squarenorm() const
    {
        return squarenorm(grad);
    }
    inline void scaleGrad(dtype scale)
    {
        grad = grad * scale;
    }
    inline void clearGrad(dtype scale)
    {
        grad = 0;
    }
    inline void updateAdagrad(double regularizationWeight, dtype adaAlpha, dtype adaEps)
    {
        grad = grad + weight * regularizationWeight;
        egrad = egrad + grad * grad;
        weight = weight - grad * adaAlpha / F<nl_sqrt>(egrad + adaEps);
    }
    /*
    inline SHAPE shape()
    {
        return weight.shape_;
    }*/
};

template<typename xpu, int N>
class Parameter: public ParameterBase<xpu, N>
{

};

template<typename xpu>
class Parameter<xpu, 1>: public ParameterBase<xpu, 1>
{
public:
    Parameter(){}
    explicit Parameter(int nOSize)
    {
        this->weight = NewTensor<xpu>(Shape2(1, nOSize), d_zero); //must add this pointer here
        this->grad = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
        this->egrad = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    }
    explicit Parameter(Tensor<xpu,2,dtype> source): Parameter(source.size(0))
    {
        //int nOSize = source.size(0);
        //int nISize = source.size(1);
        Copy(this->weight, source);
    }
};

template<typename xpu>
class Parameter<xpu,2>:public ParameterBase<xpu, 2>{
public:
    Parameter(){}
    explicit Parameter(int nOSize, int nISize)
    {
        this->weight = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
        this->grad = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
        this->egrad = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    }
    explicit Parameter(Tensor<xpu,2,dtype> source): Parameter(source.size(0), source.size(1))
    {
        //int nOSize = source.size(0);
        //int nISize = source.size(1);
        Copy(this->weight, source);
    }
};

#define Parameter1 Parameter<xpu,1>
#define Parameter2 Parameter<xpu,2>
#endif
