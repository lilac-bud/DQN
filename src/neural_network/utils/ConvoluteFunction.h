#ifndef NEURALNETWORK_CONVOLUTEFUNCTION_H
#define NEURALNETWORK_CONVOLUTEFUNCTION_H

#include <xtensor/views/xview.hpp>
#include <xtensor/generators/xbuilder.hpp>

namespace nn
{
    using Axis = int;

    const Axis height_axis = 1;
    const Axis width_axis = 2;
    const Axis channels_axis = 3;

    template<class I, class F, class S>
    auto convolute2D(const I& inputs, const F& filters, const S& outputs_shape)
    {
        auto outputs = xt::xarray<float>::from_shape(outputs_shape);

        for (std::size_t i = 0; i < outputs_shape[height_axis]; ++i)
            for (std::size_t k = 0; k < outputs_shape[width_axis]; ++k)
            {
                //conv is a part of input taken according to filters shape with a new axis added for broadcasting purposes
                auto conv = xt::view(inputs, xt::all(), xt::newaxis(), xt::range(i, i + filters.shape()[height_axis]), 
                    xt::range(k, k + filters.shape()[width_axis]));
                //after multiplying extra axes are removed, and the result is assigned to the appropriate position of the outputs
                //in the end, filters number becomes a new channels number
                xt::view(outputs, xt::all(), i, k) = xt::sum(conv * filters, { height_axis + 1,width_axis + 1,channels_axis + 1 });
            }
        return outputs;
    }
}

#endif