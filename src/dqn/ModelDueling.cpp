#include "dqn/ModelDueling.h"
#include "neural_network/layers/LayerConv2D.h"
#include "neural_network/layers/LayerMaxPooling2D.h"
#include "neural_network/layers/LayerFlatten.h"
#include "neural_network/layers/LayerDense.h"
#include "neural_network/utils/ActivationFunctions.h"

#include <xtensor/views/xview.hpp>

using namespace xt::placeholders;
using Axis = int;

dqn::ModelDueling::ModelDueling()
{
	layers_parts[ConvStatePart] = {
		.all_layers = &layers,
		.part_begin = insert_into_layers(
			std::make_unique<nn::LayerConv2D>(10, nn::KernelSize{ 3,3 }, nn::Padding::Valid, nn::Activation::Sigmoid),
			std::make_unique<nn::LayerMaxPooling2D>(nn::PoolSize{ 2,2 }),
			std::make_unique<nn::LayerConv2D>(20, nn::KernelSize{ 3,3 }, nn::Padding::Valid, nn::Activation::Sigmoid),
			std::make_unique<nn::LayerMaxPooling2D>(nn::PoolSize{ 2,2 }),
			std::make_unique<nn::LayerFlatten>() ) - layers.begin(),
		.part_end = (std::ptrdiff_t)layers.size(),
	};
	layers_parts[ConvActionsPart] = {
		.all_layers = &layers,
		.part_begin = insert_into_layers(
			std::make_unique<nn::LayerConv2D>(10, nn::KernelSize{ 3,3 }, nn::Padding::Valid, nn::Activation::Sigmoid),
			std::make_unique<nn::LayerMaxPooling2D>(nn::PoolSize{ 2,2 }),
			std::make_unique<nn::LayerConv2D>(20, nn::KernelSize{ 3,3 }, nn::Padding::Valid, nn::Activation::Sigmoid),
			std::make_unique<nn::LayerMaxPooling2D>(nn::PoolSize{ 2,2 }),
			std::make_unique<nn::LayerFlatten>() ) - layers.begin(),
		.part_end = (std::ptrdiff_t)layers.size()
	};
	layers_parts[ValuePart] = {
		.all_layers = &layers,
		.part_begin = insert_into_layers(
			std::make_unique<nn::LayerDense>(10, nn::Activation::Sigmoid),
			std::make_unique<nn::LayerDense>(1, nn::Activation::Sigmoid) ) - layers.begin(),
		.part_end = (std::ptrdiff_t)layers.size()
	};
	layers_parts[AdvantagePart] = {
		.all_layers = &layers,
		.part_begin = insert_into_layers(
			std::make_unique<nn::LayerDense>(20, nn::Activation::Sigmoid),
			std::make_unique<nn::LayerDense>(1, nn::Activation::Sigmoid) ) - layers.begin(),
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
		if (great_part == FlatGreatPart)
			shapes[ActionsBranch][Axis{ 1 }] += shapes[StateBranch][Axis{ 1 }];
		auto& great_part_names = parts_names[great_part];
		for (int branch = StateBranch; branch != BranchesTotal; branch++)
			for (auto& layer : layers_parts[great_part_names[branch]])
				layer->build(shapes[branch]);
	}
}

void dqn::ModelDueling::call_layers_part(LayersPartName layers_part_name, xt::xarray<float>& inputs, nn::Tape* tape) const
{
	for (const auto& layer : layers_parts[layers_part_name])
		layer->forward(inputs, tape);
}

void dqn::ModelDueling::get_gradient_from_layers_part(LayersPartName layers_part_name, xt::xarray<float>& outputs, xt::xarray<float>& deltas,
	nn::Tape& tape, nn::GradientMap& gradient_map) const
{
	auto& cur_layer_part = layers_parts[layers_part_name];
	for (auto layer_it = cur_layer_part.rbegin(); layer_it != cur_layer_part.rend(); layer_it++)
		(*layer_it)->backward(deltas, outputs, tape, gradient_map);
}

xt::xarray<float> dqn::ModelDueling::call_with_tape(std::array<xt::xarray<float>, 2>& inputs, nn::Tape* tape) const
{
	for (int great_part = ConvGreatPart; great_part != GreatPartsTotal; great_part++)
	{
		if (great_part == FlatGreatPart)
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

void dqn::ModelDueling::get_gradient(xt::xarray<float>& outputs, xt::xarray<float> deltas, nn::Tape& tape,
	nn::GradientMap& gradient_map) const
{
	std::array<xt::xarray<float>, BranchesTotal> branch_outputs;
	std::array<xt::xarray<float>, BranchesTotal> branch_deltas;
	branch_outputs.fill(outputs / 2);
	branch_deltas.fill(deltas);
	//we need to go backwards to get the gradient
	for (int great_part = FlatGreatPart; great_part >= ConvGreatPart; great_part--)
	{
		if (great_part == ConvGreatPart)
		{
			auto& [state_deltas, actions_deltas] = branch_deltas;
			//since actions have been concatenated during the call, the following needs to be done:
			//	1. first half of actions deltas must be added to state deltas
			//	2. only second half of actions deltas must be left
			//actions break point corresponds to state deltas size
			std::size_t actions_break_point = state_deltas.shape()[Axis{ 1 }];
			auto actions_deltas_half = xt::view(actions_deltas, xt::all(), xt::range(0, actions_break_point));
			//extra axis is temporarily added to get rid of actions deltas batch size
			state_deltas = xt::sum(xt::view(state_deltas, xt::all(), xt::newaxis()) + actions_deltas_half, { Axis{1} });
			actions_deltas = xt::view(actions_deltas, xt::all(), xt::range(actions_break_point, _));
			//the same must be done for outputs
			auto& action_outputs = branch_outputs[ActionsBranch];
			action_outputs = xt::view(action_outputs, xt::all(), xt::range(actions_break_point, _));
		}
		auto& great_part_names = parts_names[great_part];
		for (int branch = ActionsBranch; branch >= StateBranch; branch--)
			get_gradient_from_layers_part(great_part_names[branch], branch_outputs[branch], branch_deltas[branch], tape, gradient_map);
	}
}