#include "dqn/ModelDueling.h"
#include "neural_network/layers/LayerConv2D.h"
#include "neural_network/layers/LayerMaxPooling2D.h"
#include "neural_network/layers/LayerFlatten.h"
#include "neural_network/layers/LayerDense.h"
#include <xtensor/containers/xadapt.hpp>

using Axis = int;

dqn::ModelDueling::ModelDueling()
{
	layers_parts[ConvStatePart] = {
		.all_layers = &layers,
		.part_begin = layers.insert(layers.end(),{
			new nn::LayerConv2D(10, { 3,3 }, nn::Padding::Valid),
			new nn::LayerMaxPooling2D({ 2,2 }),
			new nn::LayerConv2D(20, { 3,3 }, nn::Padding::Valid),
			new nn::LayerMaxPooling2D({ 2,2 }),
			new nn::LayerFlatten }) - layers.begin(),
		.part_end = (std::ptrdiff_t)layers.size(),
	};
	layers_parts[ConvActionsPart] = {
		.all_layers = &layers,
		.part_begin = layers.insert(layers.end(),{
			new nn::LayerConv2D(10, { 3,3 }, nn::Padding::Valid),
			new nn::LayerMaxPooling2D({ 2,2 }),
			new nn::LayerConv2D(20, { 3,3 }, nn::Padding::Valid),
			new nn::LayerMaxPooling2D({ 2,2 }),
			new nn::LayerFlatten }) - layers.begin(),
		.part_end = (std::ptrdiff_t)layers.size()
	};
	layers_parts[ValuePart] = {
		.all_layers = &layers,
		.part_begin = layers.insert(layers.end(),{
			new nn::LayerDense(10), 
			new nn::LayerDense(1) }) - layers.begin(),
		.part_end = (std::ptrdiff_t)layers.size()
	};
	layers_parts[AdvantagePart] = {
		.all_layers = &layers,
		.part_begin = layers.insert(layers.end(),{
			new nn::LayerDense(20),
			new nn::LayerDense(1) }) - layers.begin(),
		.part_end = (std::ptrdiff_t)layers.size()
	};
	for (auto& part : layers_parts)
	{
		part.part_rbegin = layers.size() - part.part_end;
		part.part_rend = layers.size() - part.part_begin;
	}
}

void dqn::ModelDueling::build(std::vector<std::size_t> input_shape) const
{
	std::array<std::vector<std::size_t>, BranchesTotal> shapes;
	shapes.fill(input_shape);
	for (int great_part = ConvGreatPart; great_part != GreatPartsTotal; great_part++)
	{
		//to get advantage function actions need to be concatenated with the state
		//that's why shape is also modifyed
		if (great_part != ConvGreatPart)
			shapes[ActionsBranch][Axis{ 1 }] += shapes[StateBranch][Axis{ 1 }];
		auto& great_part_names = parts_names[great_part];
		for (int branch = StateBranch; branch != BranchesTotal; branch++)
			for (nn::Layer* layer : layers_parts[great_part_names[branch]])
				layer->build(shapes[branch]);
	}
}

void dqn::ModelDueling::call_layers_part(LayersPartName layers_part_name, xt::xarray<float>& inputs, nn::Tape* tape) const
{
	for (const nn::Layer* layer : layers_parts[layers_part_name])
		layer->forward(inputs, tape);
}

void dqn::ModelDueling::get_gradient_from_layers_part(LayersPartName layers_part_name, nn::Tape& tape, 
	nn::GradientMap& gradient_map, xt::xarray<float>& deltas) const
{
	auto& cur_layer_part = layers_parts[layers_part_name];
	for (auto layer_it = cur_layer_part.rbegin(); layer_it != cur_layer_part.rend(); layer_it++)
		(*layer_it)->backward(tape, gradient_map, deltas);
}

xt::xarray<float> dqn::ModelDueling::call_with_tape(xt::xarray<float>& state, xt::xarray<float>& actions, nn::Tape* tape) const
{
	std::array inputs{ state, actions };
	for (int great_part = ConvGreatPart; great_part != GreatPartsTotal; great_part++)
	{
		if (great_part != ConvGreatPart)
		{
			//to get advantage function actions need to be concatenated with the state
			auto& [state, actions] = inputs;
			actions = concatenate(xtuple(broadcast(state, actions.shape()), actions), Axis{ 1 });
		}
		auto& great_part_names = parts_names[great_part];
		for (int branch = StateBranch; branch != BranchesTotal; branch++)
			call_layers_part(great_part_names[branch], inputs[branch], tape);
	}
	auto&[value, advantage] = inputs;
	return value + advantage;
}

xt::xarray<xt::xarray<float>> dqn::ModelDueling::get_gradient(nn::Tape& tape, xt::xarray<float> deltas) const
{
	nn::GradientMap gradient_map;
	deltas = nn::Layer::sigmoid_derivative(deltas);
	std::array<xt::xarray<float>, BranchesTotal> branch_deltas;
	branch_deltas.fill(deltas);
	//we need to go backwards to get the gradient
	for (int great_part = FlatGreatPart; great_part >= ConvGreatPart; great_part--)
	{
		if (great_part != FlatGreatPart)
		{
			auto& [state_deltas, actions_deltas] = branch_deltas;
			//since actions have been concatenated during the call, the following needs to be done:
			//	1. first half of actions deltas must be added to state deltas
			//	2. only second half of actions deltas must be left
			//actions break point corresponds to state deltas size
			std::size_t actions_break_point = state_deltas.shape()[Axis{ 1 }];
			std::size_t actions_size = actions_deltas.shape()[Axis{ 1 }];
			auto actions_deltas_half = xt::view(actions_deltas, xt::all(), xt::range(0, actions_break_point));
			//extra axis is temporarily added to get rid of actions deltas batch size
			state_deltas = xt::sum(xt::view(state_deltas, xt::all(), xt::newaxis()) + actions_deltas_half, { Axis{1} });
			actions_deltas = xt::view(actions_deltas, xt::all(), xt::range(actions_break_point, actions_size));
		}
		auto& great_part_names = parts_names[great_part];
		for (int branch = ActionsBranch; branch >= StateBranch; branch--)
			get_gradient_from_layers_part(great_part_names[branch], tape, gradient_map, branch_deltas[branch]);
	}
	std::vector<xt::xarray<float>> gradient;
	for (auto i = gradient_map.begin(); i != gradient_map.end(); i++)
		gradient.push_back(i->second);
	return xt::adapt(gradient);
}