#include "neural_network/layers/LayerDense.h"
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/io/xio.hpp>

nn::LayerDense::LayerDense(std::size_t outputs_number)
{
	this->outputs_number = outputs_number;
	biases = xt::random::rand<float>({ outputs_number }, 0.0f, 0.1f);
}

void nn::LayerDense::build(std::vector<std::size_t>& input_shape)
{
	weights = xt::random::rand<float>({ outputs_number,  input_shape[0] }, 0.0f, 0.1f);
	input_shape = { outputs_number };
}

void nn::LayerDense::forward(xt::xarray<float>& inputs, std::map<const Layer*, xt::xarray<float>>* tape) const
{
	Layer::forward(inputs, tape);
	auto linear_res = xt::sum(weights * xt::view(inputs, xt::all(), xt::newaxis(), xt::all()), { 2 }) + biases;
	inputs = sigmoid(linear_res);
}

void nn::LayerDense::backward(std::map<const Layer*, xt::xarray<float>>& tape, std::vector<xt::xarray<float>>& gradient,
	xt::xarray<float>& deltas) const
{
	const auto& inputs = tape[this];
	auto weight_derivative = xt::sum(xt::view(xt::transpose(deltas), xt::all(), xt::newaxis(), xt::all()) * 
		xt::transpose(inputs), { 2 });
	auto biases_derivative = xt::sum(deltas, { 0 });
	gradient.push_back(weight_derivative);
	gradient.push_back(biases_derivative);
	auto res = xt::sum(xt::transpose(weights) * xt::view(deltas, xt::all(), xt::newaxis(), xt::all()), { 2 });
	deltas = res * sigmoid_derivative(inputs);
}

void nn::LayerDense::get_trainable_vars(std::vector<xt::xarray<float>*>& trainable_vars)
{
	trainable_vars.push_back(&weights);
	trainable_vars.push_back(&biases);
}

void nn::LayerDense::print_trainable_vars() const
{
	std::cout << weights << std::endl;
	std::cout << biases << std::endl;
}