#ifndef NEURALNETWORK_LAYERCONV2D_H
#define NEURALNETWORK_LAYERCONV2D_H

#include "neural_network/layers/Layer.h"
#include <xtensor/views/xview.hpp>
#include <xtensor/io/xio.hpp>

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
        xt::xarray<float> filters;
        xt::xarray<float> biases;
        std::vector<std::size_t> outputs_shape;
        std::vector<std::size_t> kernel_size;
        std::size_t filters_number;
        Padding padding;
        std::vector<std::size_t> pads;

        template<class E1, class E2, class E3>
        static auto convolute(const xt::xexpression<E1>& inputs_expr, const xt::xexpression<E2>& filters_expr,
            const xt::xexpression<E3>& outputs_expr)
        {
            auto inputs = inputs_expr.derived_cast();
            auto filters = filters_expr.derived_cast();
            auto outputs = outputs_expr.derived_cast();
            for (std::size_t i = 0; i < outputs.shape()[1]; i++)
                for (std::size_t k = 0; k < outputs.shape()[2]; k++)
                {
                    auto conv = xt::view(inputs, xt::all(), xt::newaxis(),
                        xt::range(i, i + filters.shape()[1]), xt::range(k, k + filters.shape()[2]));
                    xt::view(outputs, xt::all(), i, k) = xt::sum(conv * filters, { 2,3,4 });
                }
            return outputs;
        }

        auto pad(xt::xarray<float>& array_to_pad) const;

    public:
        LayerConv2D(std::size_t filters_number, std::vector<std::size_t> kernel_size, Padding padding);
        virtual void build(std::vector<std::size_t>& input_shape) override;
        virtual void forward(xt::xarray<float>& inputs, std::map<const Layer*, xt::xarray<float>>* tape) const override;
        virtual void backward(std::map<const Layer*, xt::xarray<float>>& tape, std::vector<xt::xarray<float>>& gradient,
            xt::xarray<float>& deltas) const override;
        virtual void get_trainable_vars(std::vector<xt::xarray<float>*>& trainable_vars) override;
        virtual void print_trainable_vars() const override;
    };
}

#endif