#ifndef NEURALNETWORK_LAYERDENSE_H
#define NEURALNETWORK_LAYERDENSE_H

#include "neural_network/layers/Layer.h"

namespace nn
{
    class LayerDense : public Layer
    {
    protected:
        xt::xarray<float> weights;
        xt::xarray<float> biases;
        std::size_t outputs_number = 0;

    public:
        LayerDense(std::size_t outputs_number);
        virtual void build(std::vector<size_t>& input_shape) override;
        virtual void forward(xt::xarray<float>& inputs, std::map<const Layer*, xt::xarray<float>>* tape) const override;
        virtual void backward(std::map<const Layer*, xt::xarray<float>>& tape, std::vector<xt::xarray<float>>& gradient,
            xt::xarray<float>& deltas) const override;
        virtual void get_trainable_vars(std::vector<xt::xarray<float>*>& trainable_vars) override;
        virtual void print_trainable_vars() const override;
    };
}

#endif