#include "neural_network/layers/LayerDense.h"
#include "neural_network/utils/ActivationFunctions.h"

#include <xtensor/generators/xrandom.hpp>
#include <xtensor/io/xio.hpp>

nn::LayerDense::LayerDense(std::size_t outputs_number, Activation activation)
{
	this->outputs_number = outputs_number;
	this->activation = activation;
	biases = xt::random::rand<float>({ outputs_number }, lower_rand_bound, upper_rand_bound);
}

void nn::LayerDense::build(std::vector<std::size_t>& input_shape)
{
	weights = xt::random::rand<float>({ outputs_number,  input_shape[input_axis] }, lower_rand_bound, upper_rand_bound);
	input_shape[input_axis] = outputs_number;
}

void nn::LayerDense::forward(xt::xarray<float>& inputs) const
{
	auto linear_res = xt::sum(weights * xt::view(inputs, xt::all(), xt::newaxis(), xt::all()), { input_axis + 1 }) + biases;
	inputs = activate(linear_res, activation);
}

void nn::LayerDense::backward(xt::xarray<float>& outputs, xt::xarray<float>& deltas, Tape& tape, GradientMap& gradient_map) const
{
	const auto& inputs = tape[this];
	deltas *= get_derivative(outputs, activation);
	auto transposed_deltas = xt::view(xt::transpose(deltas), xt::all(), xt::newaxis(), xt::all());
	auto weight_derivative = xt::sum(transposed_deltas * xt::transpose(inputs), { input_axis + 1 });
	auto biases_derivative = xt::sum(deltas, { batch_size_axis });
	gradient_map.insert({ {this, TrainableVarsType::Weights}, weight_derivative });
	gradient_map.insert({ {this, TrainableVarsType::Biases}, biases_derivative });
	deltas = xt::sum(xt::transpose(weights) * xt::view(deltas, xt::all(), xt::newaxis(), xt::all()), { input_axis + 1 });
	outputs = inputs;
}

void nn::LayerDense::get_trainable_vars(TrainableVars& trainable_vars)
{
	trainable_vars.push_back(&weights);
	trainable_vars.push_back(&biases);
}

void nn::LayerDense::get_trainable_vars(TrainableVarsMap& trainable_vars_map)
{
	trainable_vars_map.insert({ {this, TrainableVarsType::Weights}, &weights });
	trainable_vars_map.insert({ {this, TrainableVarsType::Biases}, &biases });
}

void nn::LayerDense::print_trainable_vars() const
{
	std::cout << weights << std::endl << biases << std::endl;
}