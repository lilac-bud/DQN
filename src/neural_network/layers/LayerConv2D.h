#ifndef NEURALNETWORK_LAYERCONV2D_H
#define NEURALNETWORK_LAYERCONV2D_H

#include "neural_network/layers/Layer.h"
#include <xtensor/views/xview.hpp>
#include <xtensor/io/xio.hpp>
#include <array>

namespace nn
{
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

        template<class E1, class E2, class S>
        static auto convolute(const xt::xexpression<E1>& inputs_expr, const xt::xexpression<E2>& filters_expr, const S& outputs_shape)
        {
            auto inputs = inputs_expr.derived_cast();
            auto filters = filters_expr.derived_cast();
            auto outputs = xt::xarray<float>::from_shape(outputs_shape);
            for (std::size_t i = 0; i < outputs.shape()[height_axis]; i++)
            {
                auto height_range = xt::range(i, i + filters.shape()[height_axis]);
                for (std::size_t k = 0; k < outputs.shape()[width_axis]; k++)
                {
                    //conv is a part of input taken according to filters shape with a new axis added for broadcasting purposes
                    auto conv = xt::view(inputs, xt::all(), xt::newaxis(), height_range, xt::range(k, k + filters.shape()[width_axis]));
                    //after multiplying extra axes are removed, and the result is assigned to the appropiate position of the outputs
                    //in the end, filters number becomes a new channels number
                    xt::view(outputs, xt::all(), i, k) = xt::sum(conv * filters, { height_axis + 1,width_axis + 1,channels_axis + 1 });
                }
            }
            return outputs;
        }

    public:
        LayerConv2D(std::size_t filters_number, std::array<std::size_t, 2> kernel_size, Padding padding);
        virtual void build(std::vector<std::size_t>& shape) override;
        virtual void backward(Tape& tape, GradientMap& gradient_map, xt::xarray<float>& deltas) const override;
        virtual void get_trainable_vars(TrainableVars& trainable_vars) override;
        virtual void get_trainable_vars(TrainableVarsMap& trainable_vars_map) override;
        virtual void print_trainable_vars() const override;

    private:
        virtual void forward(xt::xarray<float>& inputs) const override;
    };
}

#endif