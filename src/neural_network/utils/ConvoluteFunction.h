#ifndef NEURALNETWORK_CONVOLUTEFUNCTION_H
#define NEURALNETWORK_CONVOLUTEFUNCTION_H

#include <xtensor/views/xview.hpp>

namespace nn
{
    using Axis = int;

    template<class E1, class E2, class S>
    auto convolute(const xt::xexpression<E1>& inputs_expr, const xt::xexpression<E2>& filters_expr, const S& outputs_shape)
    {
        const Axis height_axis = 1;
        const Axis width_axis = 2;
        const Axis channels_axis = 3;

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
}

#endif