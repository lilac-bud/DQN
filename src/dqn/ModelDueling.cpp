#include "dqn/ModelDueling.h"
#include "neural_network/layers/LayerConv2D.h"
#include "neural_network/layers/LayerMaxPooling2D.h"
#include "neural_network/layers/LayerFlatten.h"
#include "neural_network/layers/LayerDense.h"
#include <xtensor/containers/xadapt.hpp>

dqn::ModelDueling::ModelDueling()
{
	conv_state_branch = {
		new nn::LayerConv2D(10, { 3,3 }, nn::Padding::Valid),
		new nn::LayerMaxPooling2D({ 2,2 }),
		new nn::LayerConv2D(20, { 3,3 }, nn::Padding::Valid),
		new nn::LayerMaxPooling2D({ 2,2 }),
		new nn::LayerFlatten };

	conv_actions_branch = {
		new nn::LayerConv2D(10, { 3,3 }, nn::Padding::Valid),
		new nn::LayerMaxPooling2D({ 2,2 }),
		new nn::LayerConv2D(20, { 3,3 }, nn::Padding::Valid),
		new nn::LayerMaxPooling2D({ 2,2 }),
		new nn::LayerFlatten };

	value_branch = { new nn::LayerDense(10), new nn::LayerDense(1) };
	advantage_branch = { new nn::LayerDense(20), new nn::LayerDense(1) };

	layers.insert(layers.end(), conv_state_branch.begin(), conv_state_branch.end());
	layers.insert(layers.end(), conv_actions_branch.begin(), conv_actions_branch.end());
	layers.insert(layers.end(), value_branch.begin(), value_branch.end());
	layers.insert(layers.end(), advantage_branch.begin(), advantage_branch.end());
}

void dqn::ModelDueling::build(std::vector<size_t> input_shape) const
{
	auto& state_shape = input_shape;
	std::vector<size_t> action_shape(input_shape);
	for (auto& layer : conv_state_branch)
		layer->build(state_shape);
	for (auto& layer : conv_actions_branch)
		layer->build(action_shape);
	action_shape[0] += state_shape[0];
	for (auto& layer : value_branch)
		layer->build(state_shape);
	for (auto& layer : advantage_branch)
		layer->build(action_shape);
}

void dqn::ModelDueling::call_branch(const std::vector<nn::Layer*>& branch, xt::xarray<float>& inputs, nn::Tape* tape) const
{
	for (const nn::Layer* layer : branch)
		layer->forward(inputs, tape);
}

void dqn::ModelDueling::get_gradient_from_branch(const std::vector<nn::Layer*>& branch, nn::Tape& tape, nn::GradientMap& gradient_map,
	xt::xarray<float>& deltas) const
{
	for (auto layer = branch.rbegin(); layer != branch.rend(); layer++)
		(*layer)->backward(tape, gradient_map, deltas);
}

xt::xarray<float> dqn::ModelDueling::call_with_tape(xt::xarray<float>& state, xt::xarray<float>& actions, nn::Tape* tape) const
{
	call_branch(conv_state_branch, state, tape);
	call_branch(conv_actions_branch, actions, tape);
	auto& value = state;
	xt::xarray<float> advantage = concatenate(xtuple(broadcast(state, actions.shape()), actions), 1);
	call_branch(value_branch, value, tape);
	call_branch(advantage_branch, advantage, tape);
	return value + advantage;
}

xt::xarray<xt::xarray<float>> dqn::ModelDueling::get_gradient(nn::Tape& tape, xt::xarray<float> deltas) const
{
	nn::GradientMap gradient_map;
	deltas = nn::Layer::sigmoid_derivative(deltas);
	auto& value_deltas = deltas;
	xt::xarray<float> advantage_deltas = deltas;

	get_gradient_from_branch(advantage_branch, tape, gradient_map, advantage_deltas);
	get_gradient_from_branch(value_branch, tape, gradient_map, value_deltas);

	xt::xarray<float> state_deltas = xt::sum(xt::view(value_deltas, xt::all(), xt::newaxis()) +
		xt::view(advantage_deltas, xt::all(), xt::range(0, value_deltas.shape()[1])), { 1 });
	xt::xarray<float> actions_deltas = xt::view(advantage_deltas, xt::all(),
		xt::range(value_deltas.shape()[1], advantage_deltas.shape()[1]));

	get_gradient_from_branch(conv_actions_branch, tape, gradient_map, actions_deltas);
	get_gradient_from_branch(conv_state_branch, tape, gradient_map, state_deltas);

	std::vector<xt::xarray<float>> gradient;
	for (auto i = gradient_map.begin(); i != gradient_map.end(); i++)
		gradient.push_back(i->second);
	return xt::adapt(gradient);
}