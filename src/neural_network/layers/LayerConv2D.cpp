#include "neural_network/layers/LayerConv2D.h"
#include <xtensor/misc/xpad.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/io/xio.hpp>

nn::LayerConv2D::LayerConv2D(std::size_t filters_number, std::array<std::size_t, 2> kernel_size, Padding padding)
{
	this->filters_number = filters_number;
	kernel_height = kernel_size[Axis{ 0 }];
	kernel_width = kernel_size[Axis{ 1 }];
	this->padding = padding;
	biases = xt::random::rand<float>({ filters_number }, lower_rand_bound, upper_rand_bound);
}

void nn::LayerConv2D::build(std::vector<std::size_t>& shape)
{
	const std::vector<std::size_t> filters_shape = { filters_number, kernel_height, kernel_width, shape[channels_axis] };
	filters = xt::random::rand<float>(filters_shape, lower_rand_bound, upper_rand_bound);
	pads.push_back({ 0,0 });
	std::size_t pad_height_axis = 0;
	std::size_t pad_width_axis = 0;
	switch (padding)
	{
	case Padding::Valid:
		pad_height_axis = kernel_height - 1;
		pad_width_axis = kernel_width - 1;
		shape[height_axis] -= pad_height_axis;
		shape[width_axis] -= pad_width_axis;
		break;
	case Padding::Same:
		pad_height_axis = (kernel_height - 1) / 2;
		pad_width_axis = (kernel_width - 1) / 2;
		break;
	default:
		break;
	}
	pads.push_back({ pad_height_axis,pad_height_axis });
	pads.push_back({ pad_width_axis,pad_width_axis });
	pads.push_back({ 0,0 });
	shape[channels_axis] = filters_number;
	outputs_shape = shape;
}

void nn::LayerConv2D::forward(xt::xarray<float>& inputs) const
{
	std::vector<std::size_t> shape(outputs_shape);
	shape[batch_size_axis] = inputs.shape()[batch_size_axis];
	auto linear_res = convolute(padding == Padding::Same ? xt::pad(inputs, pads) : inputs, filters, shape) + biases;
	inputs = sigmoid(linear_res);
}

void nn::LayerConv2D::backward(Tape& tape, GradientMap& gradient_map, xt::xarray<float>& deltas) const
{
	const auto& inputs = tape[this];

	//to get weight derivative properly the following needs to be considered
	//	1. deltas are used as filters in convolute operation
	//	2. for convolute operation to work channels axes of inputs and deltas need to allign, that is be equal
	//	3. deltas channel number is equal to filters number
	//that's why batch_size_axis and channels_axis need to be swapped for both inputs and deltas
	const auto transposed_inputs = xt::swapaxes(inputs, batch_size_axis, channels_axis);
	const auto transposed_deltas = xt::swapaxes(deltas, batch_size_axis, channels_axis);

	//in convolute operation the number of filters becomes the outputs channels number
	//but we need the channels number of inputs (which is currently in batch_size_axis) to be preserved
	//that's why the outputs shape has to get batch_size_axis and channels_axis swapped as well
	const auto& transposed_outputs_shape = xt::swapaxes(filters, batch_size_axis, channels_axis).shape();
	auto transposed_weight_derivative = convolute(transposed_inputs, transposed_deltas, transposed_outputs_shape);

	//after transposing the result of convolution we get a proper weight derivative
	auto weight_derivative = xt::swapaxes(transposed_weight_derivative, batch_size_axis, channels_axis);

	//biases are applyed per filter, and since deltas channels number is equal to filters number,
	//we only need to get rid of extra axes
	auto biases_derivative = xt::sum(deltas, { batch_size_axis, height_axis, width_axis });

	gradient_map.insert({ {this, TrainableVarsType::Weights}, weight_derivative });
	gradient_map.insert({ {this, TrainableVarsType::Biases}, biases_derivative });

	//again, convolute operation requares channels axes to allign, so to get new deltas filters need to be transposed
	auto res = convolute(xt::pad(deltas, pads), xt::swapaxes(filters, batch_size_axis, channels_axis), inputs.shape());
	deltas = res * sigmoid_derivative(inputs);
}

void nn::LayerConv2D::get_trainable_vars(TrainableVars& trainable_vars)
{
	trainable_vars.push_back(&filters);
	trainable_vars.push_back(&biases);
}

void nn::LayerConv2D::get_trainable_vars(TrainableVarsMap& trainable_vars_map)
{
	trainable_vars_map.insert({ {this, TrainableVarsType::Weights}, &filters });
	trainable_vars_map.insert({ {this, TrainableVarsType::Biases}, &biases });
}

void nn::LayerConv2D::print_trainable_vars() const
{
	std::cout << filters << std::endl;
	std::cout << biases << std::endl;
}