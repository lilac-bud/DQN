#include "neural_network/layers/LayerFlatten.h"

#include <xtensor/core/xlayout.hpp>

void nn::LayerFlatten::build(std::vector<std::size_t>& input_shape)
{
	outputs_number = 1;
	for (std::size_t shape_elem : input_shape)
		outputs_number *= shape_elem;
	input_shape = { input_shape[batch_size_axis], outputs_number };
}

void nn::LayerFlatten::forward(xt::xarray<float>& inputs) const
{
	inputs.reshape({ inputs.shape()[batch_size_axis], outputs_number });
}

void nn::LayerFlatten::backward(xt::xarray<float>& outputs, xt::xarray<float>& deltas, Tape& tape, GradientMap& gradient_map) const
{
	const auto& inputs = tape[this];
	deltas.reshape(inputs.shape());
	outputs = inputs;
}