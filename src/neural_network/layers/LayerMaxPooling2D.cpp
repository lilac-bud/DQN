#include "neural_network/layers/LayerMaxPooling2D.h"

#include <xtensor/views/xview.hpp>
#include <xtensor/views/xindex_view.hpp>

nn::LayerMaxPooling2D::LayerMaxPooling2D(PoolSize pool_size)
{
	std::tie(pool_height, pool_width) = pool_size;
}

void nn::LayerMaxPooling2D::build(std::vector<std::size_t>& input_shape)
{
	input_shape[height_axis] = (input_shape[height_axis] - 1) / pool_height + 1;
	input_shape[width_axis] = (input_shape[width_axis] - 1) / pool_width + 1;
	outputs_shape = input_shape;
}

void nn::LayerMaxPooling2D::forward(xt::xarray<float>& inputs) const
{
	std::vector<std::size_t> shape(outputs_shape);
	shape[batch_size_axis] = inputs.shape()[batch_size_axis];
	auto outputs = xt::xarray<float>::from_shape(shape);
	for (std::size_t i = 0; i < outputs_shape[height_axis]; i++)
	{
		const std::size_t inputs_i = i * pool_height;
		auto height_range = xt::range(inputs_i, inputs_i + pool_height);
		for (std::size_t k = 0; k < outputs_shape[width_axis]; k++)
		{
			std::size_t inputs_k = k * pool_width;
			//pool is a part of input taken according to pool size
			auto pool = xt::view(inputs, xt::all(), height_range, xt::range(inputs_k, inputs_k + pool_width));
			xt::view(outputs, xt::all(), i, k) = xt::amax(pool, { height_axis, width_axis });
		}
	}
	inputs = outputs;
}

void nn::LayerMaxPooling2D::backward(Tape& tape, GradientMap& gradient_map, xt::xarray<float>& deltas) const
{
	xt::xarray<float> inputs = tape[this];
	for (std::size_t i = 0; i < outputs_shape[height_axis]; i++)
	{
		const std::size_t inputs_i = i * pool_height;
		auto height_range = xt::range(inputs_i, inputs_i + pool_height);
		for (std::size_t k = 0; k < outputs_shape[width_axis]; k++)
		{
			std::size_t inputs_k = k * pool_width;
			auto pool = xt::view(inputs, xt::all(), height_range, xt::range(inputs_k, inputs_k + pool_width));
			//amax reduces the number of dimensions, but we need to preserve them for filtration
			auto max = xt::view(xt::amax(pool, { height_axis, width_axis }), xt::all(), xt::newaxis(), xt::newaxis());
			xt::filtration(pool, pool < max) = 0;
			xt::filtration(pool, pool > 0) = 1;
			pool *= xt::view(deltas, xt::all(), xt::range(i, i + 1), xt::range(k, k + 1));
		}
	}
	deltas = inputs;
}