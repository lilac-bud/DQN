#ifndef NEURALNETWORK_LAYERCONV2D_H
#define NEURALNETWORK_LAYERCONV2D_H

#include "neural_network/layers/Layer.h"

namespace nn
{
    using KernelSize = std::pair<std::size_t, std::size_t>;

    enum class Activation;

    enum class Padding
    {
        Valid,
        Same
    };

    class LayerConv2D : public Layer
    {
    protected:
        static const Axis height_axis = 1;
        static const Axis width_axis = 2;
        static const Axis channels_axis = 3;

        xt::xarray<float> filters;
        xt::xarray<float> biases;
        std::vector<std::size_t> outputs_shape;
        std::size_t kernel_height;
        std::size_t kernel_width;
        std::size_t filters_number;
        Padding padding;
        std::vector<std::vector<std::size_t>> pads;
        Activation activation;

    public:
        LayerConv2D(std::size_t filters_number, KernelSize kernel_size, Padding padding, Activation activation);
        virtual void build(std::vector<std::size_t>& shape) override;
        virtual void backward(xt::xarray<float>& outputs, xt::xarray<float>& deltas, Tape& tape, GradientMap& gradient_map) const override;
        virtual void get_trainable_vars(TrainableVars& trainable_vars) override;
        virtual void get_trainable_vars(TrainableVarsMap& trainable_vars_map) override;
        virtual void print_trainable_vars() const override;

    private:
        virtual void forward(xt::xarray<float>& inputs) const override;
    };
}

#endif