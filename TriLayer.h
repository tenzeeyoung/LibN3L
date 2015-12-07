/*
 * TriLayer.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_TriLayer_H_
#define SRC_TriLayer_H_
#include "tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"
#include "Parameter.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class TriLayer {

public:

    Parameter2 _param1;
    Parameter2 _param2;
    Parameter2 _param3;
    Parameter2 _paramB;

    bool _useB;

    int _funcType; // 0: tanh, 1: sigmod, 2: f(x)=x, 3: exp

public:
    TriLayer() {
    }

    inline void initial(int nOSize, int nISize1, int nISize2, int nISize3, bool bUseB = true, int seed = 0, int funcType = 0) {
        dtype bound = sqrt(6.0 / (nOSize + nISize1 + nISize2 + nISize3 + 1));
        //dtype bound = 0.01;

        _param1 = Parameter2(nOSize, nISize1);
        _param2 = Parameter2(nOSize, nISize2);
        _param3 = Parameter2(nOSize, nISize3);

        _param1.random(bound, seed);
        _param2.random(bound, seed+1);
        _param3.random(bound, seed+2);
        if(bUseB)
        {
            _paramB = Parameter1(nOSize);
            _paramB.random(bound, seed+3);
        }
        _useB = bUseB;
        _funcType = funcType;
    }

    inline void initial(Tensor<xpu, 2, dtype> W1, Tensor<xpu, 2, dtype> W2, Tensor<xpu, 2, dtype> W3, Tensor<xpu, 2, dtype> b, bool bUseB = true,
                        int funcType = 0) {
        _param1 = Parameter2(W1);
        _param2 = Parameter2(W2);
        _param3 = Parameter2(W3);

        if (bUseB)
            _paramB = Parameter1(b);
        _useB = bUseB;
        _funcType = funcType;
    }

    inline void release() {
        _param1.release();
        _param2.release();
        _param3.release();
        if(_useB)
            _paramB.release();
    }

    virtual ~TriLayer() {
        // TODO Auto-generated destructor stub
    }

    inline dtype squarenormAll() {
        dtype result = _param1.squarenorm();
        result += _param2.squarenorm();
        result += _param3.squarenorm();
        if (_useB) {
            result += _paramB.squarenorm();
        }

        return result;
    }

    inline void scaleGrad(dtype scale) {
        _param1.scaleGrad( scale );
        _param2.scaleGrad( scale );
        _param3.scaleGrad( scale );
        if (_useB) {
            _paramB.scaleGrad(scale);
        }
    }

public:
    inline void ComputeForwardScore(Tensor<xpu, 2, dtype> x1, Tensor<xpu, 2, dtype> x2, Tensor<xpu, 2, dtype> x3, Tensor<xpu, 2, dtype> y) {
        y = dot(x1, _param1.weight.T());
        y += dot(x2, _param2.weight.T());
        y += dot(x3, _param3.weight.T());
        if (_useB)
            y = y + _paramB.weight;
        if (_funcType == 0)
            y = F<nl_tanh>(y);
        else if (_funcType == 1)
            y = F<nl_sigmoid>(y);
        else if (_funcType == 3)
            y = F<nl_exp>(y);
    }

    inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x1, Tensor<xpu, 3, dtype> x2, Tensor<xpu, 3, dtype> x3, Tensor<xpu, 3, dtype> y) {
        int seq_size = y.size(0);

        for (int id = 0; id < seq_size; id++) {
            y[id] = dot(x1[id], _param1.weight.T());
            y[id] += dot(x2[id], _param2.weight.T());
            y[id] += dot(x3[id], _param3.weight.T());
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

    inline void ComputeForwardScore(const std::vector<Tensor<xpu, 2, dtype> > &x1, const std::vector<Tensor<xpu, 2, dtype> > &x2,
                                    const std::vector<Tensor<xpu, 2, dtype> > &x3, std::vector<Tensor<xpu, 2, dtype> > &y) {
        int seq_size = y.size();

        for (int id = 0; id < seq_size; id++) {
            y[id] = dot(x1[id], _param1.weight.T());
            y[id] += dot(x2[id], _param2.weight.T());
            y[id] += dot(x3[id], _param3.weight.T());
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
    inline void ComputeBackwardLoss(Tensor<xpu, 2, dtype> x1, Tensor<xpu, 2, dtype> x2, Tensor<xpu, 2, dtype> x3, Tensor<xpu, 2, dtype> y,
                                    Tensor<xpu, 2, dtype> ly, Tensor<xpu, 2, dtype> lx1, Tensor<xpu, 2, dtype> lx2, Tensor<xpu, 2, dtype> lx3, bool bclear = false) {
        //_gradW
        Tensor<xpu, 2, dtype> deri_yx(Shape2(y.size(0), y.size(1))), cly(Shape2(y.size(0), y.size(1)));
        AllocSpace(&deri_yx);
        AllocSpace(&cly);

        if(bclear) {
            lx1 = 0.0;
            lx2 = 0.0;
            lx3 = 0.0;
        }
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
        _param1.grad += dot(cly.T(), x1);
        _param2.grad += dot(cly.T(), x2);
        _param3.grad += dot(cly.T(), x3);

        //_paramB.grad
        if (_useB)
            _paramB.grad += cly;

        //lx
        lx1 += dot(cly, _param1.weight);
        lx2 += dot(cly, _param2.weight);
        lx3 += dot(cly, _param3.weight);

        FreeSpace(&deri_yx);
        FreeSpace(&cly);
    }


    //please allocate the memory outside here
    inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x1, Tensor<xpu, 3, dtype> x2, Tensor<xpu, 3, dtype> x3, Tensor<xpu, 3, dtype> y,
                                    Tensor<xpu, 3, dtype> ly, Tensor<xpu, 3, dtype> lx1, Tensor<xpu, 3, dtype> lx2, Tensor<xpu, 3, dtype> lx3, bool bclear = false) {
        int seq_size = y.size(0);
        int y_dim1 = y.size(1), y_dim2 = y.size(2);
        //_gradW
        Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
        AllocSpace(&deri_yx);
        AllocSpace(&cly);
        if(bclear) {
            lx1 = 0.0;
            lx2 = 0.0;
            lx3 = 0.0;
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
            _param1.grad += dot(cly.T(), x1[id]);
            _param2.grad += dot(cly.T(), x2[id]);
            _param3.grad += dot(cly.T(), x3[id]);

            //_paramB.grad
            if (_useB)
                _paramB.grad += cly;

            //lx
            lx1[id] += dot(cly, _param1.weight);
            lx2[id] += dot(cly, _param2.weight);
            lx3[id] += dot(cly, _param3.weight);
        }

        FreeSpace(&deri_yx);
        FreeSpace(&cly);
    }


    //please allocate the memory outside here
    inline void ComputeBackwardLoss(const std::vector<Tensor<xpu, 2, dtype> > &x1, const std::vector<Tensor<xpu, 2, dtype> > &x2,
                                    const std::vector<Tensor<xpu, 2, dtype> > &x3, const std::vector<Tensor<xpu, 2, dtype> > &y,
                                    const std::vector<Tensor<xpu, 2, dtype> > &ly, std::vector<Tensor<xpu, 2, dtype> > &lx1,
                                    std::vector<Tensor<xpu, 2, dtype> > &lx2, std::vector<Tensor<xpu, 2, dtype> > &lx3, bool bclear = false) {
        int seq_size = y.size();
        assert(seq_size > 0);
        int y_dim1 = y[0].size(0), y_dim2 = y[0].size(1);
        //_gradW
        Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
        AllocSpace(&deri_yx);
        AllocSpace(&cly);
        if(bclear) {
            for (int id = 0; id < seq_size; id++) {
                lx1[id] = 0.0;
                lx2[id] = 0.0;
                lx3[id] = 0.0;
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
            _param1.grad += dot(cly.T(), x1[id]);
            _param2.grad += dot(cly.T(), x2[id]);
            _param3.grad += dot(cly.T(), x3[id]);

            //_paramB.grad
            if (_useB)
                _paramB.grad += cly;

            //lx
            lx1[id] += dot(cly, _param1.weight);
            lx2[id] += dot(cly, _param2.weight);
            lx3[id] += dot(cly, _param3.weight);
        }

        FreeSpace(&deri_yx);
        FreeSpace(&cly);
    }

    inline void randomprint(int num) {
        static int nOSize, nISize1, nISize2, nISize3;
        nOSize = _param1.size(0);
        nISize1 = _param1.size(1);
        nISize2 = _param2.size(1);
        nISize3 = _param3.size(1);
        int count = 0;
        while (count < num) {
            int idx1 = rand() % nOSize;
            int idy1 = rand() % nISize1;
            int idx2 = rand() % nOSize;
            int idy2 = rand() % nISize2;
            int idx3 = rand() % nOSize;
            int idy3 = rand() % nISize3;

            std::cout << "_param1[" << idx1 << "," << idy1 << "]=" << _param1.weight[idx1][idy1] << " ";
            std::cout << "_param2[" << idx2 << "," << idy2 << "]=" << _param2.weight[idx2][idy2] << " ";
            std::cout << "_param3[" << idx3 << "," << idy3 << "]=" << _param3.weight[idx3][idy3] << " ";

            if (_useB) {
                int idz = rand() % nOSize;
                std::cout << "_paramB[0][" << idz << "]=" << _paramB[0][idz] << " ";
            }
            count++;
        }

        std::cout << std::endl;
    }

    inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
        _param1.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
        _param2.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
        _param3.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);

        if (_useB) {
            _paramB.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
        }
        clearGrad();
    }

    inline void clearGrad() {
        _param1.clearGrad();
        _param2.clearGrad();
        _param3.clearGrad();
        if (_useB)
            _paramB.clearGrad();
    }
};

#endif /* SRC_TriLayer_H_ */
