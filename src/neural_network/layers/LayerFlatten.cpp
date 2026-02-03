#include "neural_network/layers/LayerFlatten.h"
#include <xtensor/core/xlayout.hpp>

void nn::LayerFlatten::build(std::vector<std::size_t>& input_shape)
{
	outputs_number = 1;
	for (std::size_t shape_elem : input_shape)
		outputs_number *= shape_elem;
	input_shape = { outputs_number };
}

void nn::LayerFlatten::forward(xt::xarray<float>& inputs, std::map<const Layer*, xt::xarray<float>>* tape) const
{
	Layer::forward(inputs, tape);
	inputs.reshape({ inputs.shape()[0], outputs_number });
}

void nn::LayerFlatten::backward(std::map<const Layer*, xt::xarray<float>>& tape, std::vector<xt::xarray<float>>& gradient,
	xt::xarray<float>& deltas) const
{
	deltas.reshape(tape[this].shape());
}