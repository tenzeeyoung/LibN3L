/*
 * BiLayer.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_paramBiLayer_H_
#define SRC_paramBiLayer_H_
#include "tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"
#include "Parameter.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class BiLayer {

public:

    Parameter2 _paramL;
    Parameter2 _paramR;
    Parameter1 _paramB;

    bool _useB;

    int _funcType; // 0: tanh, 1: sigmod, 2: f(x)=x, 3: exp

public:
    BiLayer() {
    }

    inline void initial(int nOSize, int nLISize, int nRISize, bool bUseB = true, int seed = 0, int funcType = 0) {
        dtype bound = sqrt(6.0 / (nOSize + nLISize + nRISize + 1));
        //dtype bound = 0.01;

        _paramL = Parameter2(nOSize, nLISize);
        _paramR = Parameter2(nOSize, nRISize);

        _paramL.random(bound, seed);
        _paramR.random(bound, seed+1);

        if( bUseB)
        {
            _paramB = Parameter1(nOSize);
            _paramB.random(bound, seed+2);
        }
        _useB = bUseB;
        _funcType = funcType;
    }

    inline void initial(Tensor<xpu, 2, dtype> WL, Tensor<xpu, 2, dtype> WR, Tensor<xpu, 2, dtype> b, bool bUseB = true, int funcType = 0) {

        _paramL = Parameter2(WL);
        _paramR = Parameter2(WR);

        if (bUseB)
            _paramB = Parameter2(b);

        _useB = bUseB;
        _funcType = funcType;
    }


    inline void initial(Tensor<xpu, 2, dtype> WL, Tensor<xpu, 2, dtype> WR, int funcType = 0) {

        _paramL = Parameter2(WL);
        _paramR = Parameter2(WR);

        _useB = false;
        _funcType = funcType;
    }

    inline void release() {
        _paramL.release();
        _paramR.release();
        if(_useB)
            _paramB.release();
    }

    virtual ~BiLayer() {
        // TODO Auto-generated destructor stub
    }

    inline dtype squarenormAll() {
        dtype result = _paramL.squarenorm();
        result += _paramR.squarenorm();
        if (_useB) {
            result += _paramB.squarenorm();
        }
        return result;
    }

    inline void scaleGrad(dtype scale) {
        _paramL.scaleGrad(scale);
        _paramR.scaleGrad(scale);
        if (_useB) {
            _paramB.scaleGrad(scale);
        }
    }

public:
    inline void ComputeForwardScore(Tensor<xpu, 2, dtype> xl, Tensor<xpu, 2, dtype> xr, Tensor<xpu, 2, dtype> y) {
        y = dot(xl, _paramL.weight.T());
        y += dot(xr, _paramR.weight.T());
        if (_useB)
            y = y + _paramB.weight;
        if (_funcType == 0)
            y = F<nl_tanh>(y);
        else if (_funcType == 1)
            y = F<nl_sigmoid>(y);
        else if (_funcType == 3)
            y = F<nl_exp>(y);
    }


    inline void ComputeForwardScore(Tensor<xpu, 3, dtype> xl, Tensor<xpu, 3, dtype> xr, Tensor<xpu, 3, dtype> y) {
        int seq_size = y.size(0);
        for(int id = 0; id < seq_size; id++) {
            y[id] = dot(xl[id], _paramL.weight.T());
            y[id] += dot(xr[id], _paramR.weight.T());
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

    inline void ComputeForwardScore(const std::vector<Tensor<xpu, 2, dtype> >& xl, const std::vector<Tensor<xpu, 2, dtype> >& xr,
                                    std::vector<Tensor<xpu, 2, dtype> > &y) {
        int seq_size = y.size();
        for(int id = 0; id < seq_size; id++) {
            y[id] = dot(xl[id], _paramL.weight.T());
            y[id] += dot(xr[id], _paramR.weight.T());
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
    inline void ComputeBackwardLoss(Tensor<xpu, 2, dtype> xl, Tensor<xpu, 2, dtype> xr, Tensor<xpu, 2, dtype> y, Tensor<xpu, 2, dtype> ly,
                                    Tensor<xpu, 2, dtype> lxl, Tensor<xpu, 2, dtype> lxr, bool bclear = false) {
        //_gradW
        Tensor<xpu, 2, dtype> deri_yx(Shape2(y.size(0), y.size(1))), cly(Shape2(y.size(0), y.size(1)));
        AllocSpace(&deri_yx);
        AllocSpace(&cly);
        if(bclear) {
            lxl = 0.0;
            lxr = 0.0;
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
        _paramL.grad += dot(cly.T(), xl);
        _paramR.grad += dot(cly.T(), xr);

        //_paramB.grad
        if (_useB)
            _paramB.grad += cly;

        //lx
        lxl += dot(cly, _paramL.weight);
        lxr += dot(cly, _paramR.weight);

        FreeSpace(&deri_yx);
        FreeSpace(&cly);
    }


    //please allocate the memory outside here
    inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> xl, Tensor<xpu, 3, dtype> xr, Tensor<xpu, 3, dtype> y, Tensor<xpu, 3, dtype> ly,
                                    Tensor<xpu, 3, dtype> lxl, Tensor<xpu, 3, dtype> lxr, bool bclear = false) {
        int seq_size = y.size(0);
        int y_dim1 = y.size(1), y_dim2 = y.size(2);
        //_gradW
        Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
        AllocSpace(&deri_yx);
        AllocSpace(&cly);

        if(bclear) {
            lxl = 0.0;
            lxr = 0.0;
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
            _paramL.grad += dot(cly.T(), xl[id]);
            _paramR.grad += dot(cly.T(), xr[id]);

            //_paramB.grad
            if (_useB)
                _paramB.grad += cly;

            //lx
            lxl[id] += dot(cly, _paramL.weight);
            lxr[id] += dot(cly, _paramR.weight);
        }

        FreeSpace(&deri_yx);
        FreeSpace(&cly);
    }

    inline void ComputeBackwardLoss(const std::vector<Tensor<xpu, 2, dtype> > &xl, const std::vector<Tensor<xpu, 2, dtype> > &xr,
                                    const std::vector<Tensor<xpu, 2, dtype> > &y, const std::vector<Tensor<xpu, 2, dtype> > &ly,
                                    std::vector<Tensor<xpu, 2, dtype> > &lxl, std::vector<Tensor<xpu, 2, dtype> > &lxr, bool bclear = false) {
        int seq_size = y.size();
        assert(seq_size > 0);
        int y_dim1 = y[0].size(0), y_dim2 = y[0].size(1);
        //_gradW
        Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
        AllocSpace(&deri_yx);
        AllocSpace(&cly);

        if(bclear) {
            for (int id = 0; id < seq_size; id++) {
                lxl[id] = 0.0;
                lxr[id] = 0.0;
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
            _paramL.grad += dot(cly.T(), xl[id]);
            _paramR.grad += dot(cly.T(), xr[id]);

            //_paramB.grad
            if (_useB)
                _paramB.grad += cly;

            //lx
            lxl[id] += dot(cly, _paramL.weight);
            lxr[id] += dot(cly, _paramR.weight);
        }

        FreeSpace(&deri_yx);
        FreeSpace(&cly);
    }

    inline void randomprint(int num) {
        static int nOSize, nLISize, nRISize;
        nOSize = _paramL.size(0);
        nLISize = _paramL.size(1);
        nRISize = _paramR.size(1);
        int count = 0;
        while (count < num) {
            int idxl = rand() % nOSize;
            int idyl = rand() % nLISize;
            int idxr = rand() % nOSize;
            int idyr = rand() % nRISize;

            std::cout << "_paramL[" << idxl << "," << idyl << "]=" << _paramL.weight[idxl][idyl] << " ";
            std::cout << "_paramR[" << idxr << "," << idyr << "]=" << _paramR.weight[idxr][idyr] << " ";

            if (_useB) {
                int idz = rand() % nOSize;
                std::cout << "_paramB[0][" << idz << "]=" << _paramB.weight[0][idz] << " ";
            }
            count++;
        }

        std::cout << std::endl;
    }

    inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
        _paramL.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
        _paramR.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);


        if (_useB) {
            _paramB.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
        }
        clearGrad();
    }

    inline void clearGrad() {
        _paramL.clearGrad() ;
        _paramR.clearGrad() ;
        if (_useB)
            _paramB.clearGrad();
    }
};

#endif /* SRC_paramBiLayer_H_ */
