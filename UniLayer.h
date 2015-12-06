/*
 * UniLayer.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_UniLayer_H_
#define SRC_UniLayer_H_
#include "tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"
#include "Parameter.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//#define DIM2 xpu, 2, dtype

template<typename xpu>
class UniLayer {

public:

  Parameter2 _paramW;
  Parameter2 _paramB;

  bool _useB;

  int _funcType; // 0: tanh, 1: sigmod, 2: f(x)=x, 3: exp

public:
  UniLayer() {
  }

  inline void initial(int nOSize, int nISize, bool bUseB = true, int seed = 0, int funcType = 0) {
    dtype bound = sqrt(6.0 / (nOSize + nISize + 1));
    //dtype bound = 0.01;

    _paramW = Parameter2(nOSize, nISize);
    _paramW.random(bound, seed);

    if( bUseB )
    {   
        _paramB = Parameter1(nOSize);
        _paramB.random(bound, seed + 1);
    }

    _useB = bUseB;
    _funcType = funcType;
  }

  inline void initial(Tensor<xpu, 2, dtype> W, Tensor<xpu, 2, dtype> b, bool bUseB = true, int funcType = 0) {
    /*static int nOSize, nISize;
    nOSize = W.size(0);
    nISize = W.size(1);*/

    _paramW = Parameter2(W);

    if (bUseB)
      _paramB = Parameter1(b);

    _useB = bUseB;
    _funcType = funcType;
  }

  inline void initial(Tensor<xpu, 2, dtype> W,  int funcType = 0) {
    static int nOSize, nISize;
    nOSize = W.size(0);
    //nISize = W.size(1);

    _paramW = Parameter2(W);

    //_paramB = Parameter1(nOSize);

    _useB = false;
    _funcType = funcType;
  }
  inline void release() {
    _paramW.release();
    if(_useB)
        _paramB.release();
  }

  virtual ~UniLayer() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    dtype result = _paramW.squarenorm();

    if (_useB) {
      result += _paramB.squarenorm();
    }

    return result;
  }

  inline void scaleGrad(dtype scale) {
    _paramW.scaleGrad( scale );
    if (_useB) {
      _paramB.scaleGrad( scale );
    }
  }

public:
  inline void ComputeForwardScore(Tensor<xpu, 2, dtype> x, Tensor<xpu, 2, dtype> y) {
    y = dot(x, _paramW.weight.T());
    if (_useB)
      y = y + _paramB.weight;
    if (_funcType == 0)
      y = F<nl_tanh>(y);
    else if (_funcType == 1)
      y = F<nl_sigmoid>(y);
    else if (_funcType == 3)
      y = F<nl_exp>(y);
  }

  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> y) {
    int seq_size = y.size(0);
    for (int id = 0; id < seq_size; id++) {
      y[id] = dot(x[id], _paramW.weight.T());
      if (_useB)
        y[id] = y[id] + _paramB.weight;
      if (_funcType == 0)
        y[id] = F<nl_tanh>(y[id]);
      else if (_funcType == 1)
        y[id] = F<nl_sigmoid>(y[id]);
      else if (_funcType == 3)
        y[id] = F<nl_exp>(y[id]);
    }
  }

  inline void ComputeForwardScore(const std::vector<Tensor<xpu, 2, dtype> > &x, std::vector<Tensor<xpu, 2, dtype> > &y) {
    int seq_size = y.size();
    for (int id = 0; id < seq_size; id++) {
      y[id] = dot(x[id], _paramW.weight.T());
      if (_useB)
        y[id] = y[id] + _paramB.weight;
      if (_funcType == 0)
        y[id] = F<nl_tanh>(y[id]);
      else if (_funcType == 1)
        y[id] = F<nl_sigmoid>(y[id]);
      else if (_funcType == 3)
        y[id] = F<nl_exp>(y[id]);
    }
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 2, dtype> x, Tensor<xpu, 2, dtype> y, Tensor<xpu, 2, dtype> ly, Tensor<xpu, 2, dtype> lx, bool bclear = false) {
    //_gradW
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y.size(0), y.size(1))), cly(Shape2(y.size(0), y.size(1)));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

    if (bclear)
      lx = 0.0;
    if (_funcType == 0) {
      deri_yx = F<nl_dtanh>(y);
      cly = ly * deri_yx;
    } else if (_funcType == 1) {
      deri_yx = F<nl_dsigmoid>(y);
      cly = ly * deri_yx;
    } else if (_funcType == 3) {
      cly = ly * y;
    } else {
      //cly = ly;
      Copy(cly, ly);
    }
    //_gradW
    _paramW.grad += dot(cly.T(), x);

    //_gradb
    if (_useB)
      _paramB.grad += cly;

    //lx
    lx += dot(cly, _paramW.weight);

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> y, Tensor<xpu, 3, dtype> ly, Tensor<xpu, 3, dtype> lx, bool bclear = false) {
    //_gradW
    int seq_size = y.size(0);
    int y_dim1 = y.size(1), y_dim2 = y.size(2);
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

    if (bclear)
      lx = 0.0;
    for (int id = 0; id < seq_size; id++) {
      if (_funcType == 0) {
        deri_yx = F<nl_dtanh>(y[id]);
        cly = ly[id] * deri_yx;
      } else if (_funcType == 1) {
        deri_yx = F<nl_dsigmoid>(y[id]);
        cly = ly[id] * deri_yx;
      } else if (_funcType == 3) {
        cly = ly[id] * y[id];
      } else {
        //cly = ly;
        Copy(cly, ly[id]);
      }
      //_gradW
      _paramW.grad += dot(cly.T(), x[id]);

      //_gradb
      if (_useB)
        _paramW.grad += cly;

      //lx
      lx[id] += dot(cly, _paramW.weight);
    }

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(const std::vector<Tensor<xpu, 2, dtype> > &x, const std::vector<Tensor<xpu, 2, dtype> > &y,
      const std::vector<Tensor<xpu, 2, dtype> > &ly, std::vector<Tensor<xpu, 2, dtype> > &lx, bool bclear = false) {
    //_gradW
    int seq_size = y.size();
    assert(seq_size > 0);
    int y_dim1 = y[0].size(0), y_dim2 = y[0].size(1);
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

    if(bclear) {
      for (int id = 0; id < seq_size; id++) {
        lx[id] = 0.0;
      }
    }
    for (int id = 0; id < seq_size; id++) {
      if (_funcType == 0) {
        deri_yx = F<nl_dtanh>(y[id]);
        cly = ly[id] * deri_yx;
      } else if (_funcType == 1) {
        deri_yx = F<nl_dsigmoid>(y[id]);
        cly = ly[id] * deri_yx;
      } else if (_funcType == 3) {
        cly = ly[id] * y[id];
      } else {
        //cly = ly;
        Copy(cly, ly[id]);
      }
      //_gradW
      _paramW.grad += dot(cly.T(), x[id]);

      //_gradb
      if (_useB)
        _paramW.grad += cly;

      //lx
      lx[id] += dot(cly, _paramW.weight);
    }

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }

  inline void randomprint(int num) {
    static int nOSize, nISize;
    nOSize = _paramW.size(0);
    nISize = _paramW.size(1);
    int count = 0;
    while (count < num) {
      int idx = rand() % nOSize;
      int idy = rand() % nISize;

      std::cout << "_paramW[" << idx << "," << idy << "]=" << _paramW.weight[idx][idy] << " ";

      if (_useB) {
        int idz = rand() % nOSize;
        std::cout << "_paramB[0][" << idz << "]=" << _paramB.weight[0][idz] << " ";
      }
      count++;
    }

    std::cout << std::endl;
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {

    _paramW.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);

    if (_useB) {
        _paramB.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    }
    clearGrad();
  }

  inline void clearGrad() {
    _paramW.clearGrad();
    if (_useB)
      _paramB.clearGrad();
  }
};

#endif /* SRC_UniLayer_H_ */
