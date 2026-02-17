#ifndef NEURALNETWORK_LAYERDENSE_H
#define NEURALNETWORK_LAYERDENSE_H

#include "neural_network/layers/Layer.h"

namespace nn
{
    class LayerDense : public Layer
    {
    protected:
        static const Axis input_axis = 1;

        xt::xarray<float> weights;
        xt::xarray<float> biases;
        std::size_t outputs_number = 0;

    public:
        LayerDense(std::size_t outputs_number);
        virtual void build(std::vector<size_t>& input_shape) override; 
        virtual void backward(Tape& tape, GradientMap& gradient_map, xt::xarray<float>& deltas) const override;
        virtual void get_trainable_vars(TrainableVars& trainable_vars) override;
        virtual void get_trainable_vars(TrainableVarsMap& trainable_vars_map) override;
        virtual void print_trainable_vars() const override;

    private:
        virtual void forward(xt::xarray<float>& inputs) const override;
    };
}

#endif