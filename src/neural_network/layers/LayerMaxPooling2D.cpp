#include "neural_network/layers/LayerMaxPooling2D.h"
#include "neural_network/utils/PoolFunctions.h"

nn::LayerMaxPooling2D::LayerMaxPooling2D(PoolSize pool_size) : pool_size(pool_size) {}

void nn::LayerMaxPooling2D::build(std::vector<std::size_t>& input_shape)
{
	input_shape[height_axis] = (input_shape[height_axis] - 1) / pool_size.first + 1;
	input_shape[width_axis] = (input_shape[width_axis] - 1) / pool_size.second + 1;
	outputs_shape = input_shape;
}

void nn::LayerMaxPooling2D::forward(xt::xarray<float>& inputs) const
{
	std::vector<std::size_t> shape(outputs_shape);
	shape[batch_size_axis] = inputs.shape()[batch_size_axis];
	inputs = maxpool2D(inputs, shape, pool_size);
}

void nn::LayerMaxPooling2D::backward(xt::xarray<float>& outputs, xt::xarray<float>& deltas, Tape& tape, GradientMap& gradient_map) const
{
	const auto& inputs = tape[this];
	deltas = unmaxpool2D(inputs, outputs, deltas, pool_size);
	outputs = inputs;
}