#include "neural_network/layers/LayerMaxPooling2D.h"
#include <xtensor/views/xview.hpp>
#include <xtensor/views/xindex_view.hpp>

nn::LayerMaxPooling2D::LayerMaxPooling2D(std::vector<std::size_t> pool_size)
{
	this->pool_size = pool_size;
}

void nn::LayerMaxPooling2D::build(std::vector<std::size_t>& input_shape)
{
	input_shape[0] = (input_shape[0] - 1) / pool_size[0] + 1;
	input_shape[1] = (input_shape[1] - 1) / pool_size[1] + 1;
	outputs_shape = input_shape;
}

void nn::LayerMaxPooling2D::forward(xt::xarray<float>& inputs, std::map<const Layer*, xt::xarray<float>>* tape) const
{
	Layer::forward(inputs, tape);
	std::vector<std::size_t> shape(outputs_shape);
	shape.insert(shape.begin(), inputs.shape()[0]);
	auto outputs = xt::xarray<float>::from_shape(shape);
	for (std::size_t i = 0; i < outputs_shape[0]; i++)
		for (std::size_t k = 0; k < outputs_shape[1]; k++)
		{
			auto pool = xt::view(inputs, xt::all(), xt::range(i * pool_size[0], i * pool_size[0] + pool_size[0]),
				xt::range(k * pool_size[1], k * pool_size[1] + pool_size[1]));
			xt::view(outputs, xt::all(), i, k, xt::all()) = xt::amax(pool, { 1,2 });
		}
	inputs = outputs;
}

void nn::LayerMaxPooling2D::backward(std::map<const Layer*, xt::xarray<float>>& tape, std::vector<xt::xarray<float>>& gradient,
	xt::xarray<float>& deltas) const
{
	xt::xarray<float> inputs = tape[this];
	for (std::size_t i = 0; i < outputs_shape[0]; i++)
		for (std::size_t k = 0; k < outputs_shape[1]; k++)
		{
			auto pool = xt::view(inputs, xt::all(), xt::range(i * pool_size[0], i * pool_size[0] + pool_size[0]),
				xt::range(k * pool_size[1], k * pool_size[1] + pool_size[1]));
			xt::filtration(pool, pool < xt::view(xt::amax(pool, { 1,2 }), xt::all(), xt::newaxis(), xt::newaxis())) = 0;
			xt::filtration(pool, pool > 0) = 1;
			pool *= xt::view(deltas, xt::all(), i, k, xt::newaxis(), xt::newaxis());
		}
	deltas = inputs;
}