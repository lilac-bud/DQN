#include "neural_network/layers/LayerConv2D.h"
#include <xtensor/misc/xpad.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/io/xio.hpp>

nn::LayerConv2D::LayerConv2D(std::size_t filters_number, std::vector<std::size_t> kernel_size, Padding padding)
{
	this->filters_number = filters_number;
	this->kernel_size = kernel_size;
	this->padding = padding;
	biases = xt::random::rand<float>({ filters_number }, 0.0f, 0.1f);
}

void nn::LayerConv2D::build(std::vector<std::size_t>& input_shape)
{
	filters = xt::random::rand<float>({ filters_number, kernel_size[0], kernel_size[1], input_shape[2] }, 0.0f, 0.1f);
	switch (padding)
	{
	case Padding::Valid:
		pads.push_back(kernel_size[0] - 1);
		pads.push_back(kernel_size[1] - 1);
		input_shape[0] -= pads[0];
		input_shape[1] -= pads[1];
		break;
	case Padding::Same:
		pads.push_back((kernel_size[0] - 1) / 2);
		pads.push_back((kernel_size[1] - 1) / 2);
		break;
	default:
		break;
	}
	input_shape[2] = filters_number;
	outputs_shape = input_shape;
}

auto nn::LayerConv2D::pad(xt::xarray<float>& array_to_pad) const
{
	return xt::pad(array_to_pad, { {0,0}, {pads[0],pads[0]}, {pads[1],pads[1]}, {0,0} });
}

void nn::LayerConv2D::forward(xt::xarray<float>& inputs) const
{
	std::vector<std::size_t> shape(outputs_shape);
	shape.insert(shape.begin(), inputs.shape()[0]);
	auto linear_res = convolute(padding == Padding::Same ? pad(inputs) : inputs,
		filters, xt::xarray<float>::from_shape(shape)) + biases;
	inputs = sigmoid(linear_res);
}

void nn::LayerConv2D::backward(std::map<const Layer*, xt::xarray<float>>& tape, std::vector<xt::xarray<float>>& gradient,
	xt::xarray<float>& deltas) const
{
	const auto& inputs = tape[this];
	auto weight_derivative = xt::swapaxes(convolute(xt::swapaxes(inputs, 0, 3), xt::swapaxes(deltas, 0, 3),
		xt::xarray<float>::from_shape(xt::swapaxes(filters, 0, 3).shape())), 0, 3);
	auto biases_derivative = xt::sum(deltas, { 0,1,2 });
	gradient.push_back(weight_derivative);
	gradient.push_back(biases_derivative);
	auto res = convolute(pad(deltas), xt::swapaxes(filters, 0, 3),
		xt::xarray<float>::from_shape(inputs.shape()));
	deltas = res * sigmoid_derivative(inputs);
}

void nn::LayerConv2D::get_trainable_vars(std::vector<xt::xarray<float>*>& trainable_vars)
{
	trainable_vars.push_back(&filters);
	trainable_vars.push_back(&biases);
}

void nn::LayerConv2D::print_trainable_vars() const
{
	std::cout << filters << std::endl;
	std::cout << biases << std::endl;
}