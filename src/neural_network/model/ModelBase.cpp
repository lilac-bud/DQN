#include "neural_network/model/ModelBase.h"
#include "neural_network/layers/Layer.h"

#include <xtensor/io/xjson.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <fstream>

nn::ModelBase::~ModelBase() = default;

void nn::ModelBase::build(std::vector<std::size_t> input_shape) const
{
	for (auto& layer : layers)
		layer->build(input_shape);
}

xt::xarray<xt::xarray<float>> nn::ModelBase::get_gradient(xt::xarray<float> outputs, Tape& tape) const
{
	nn::GradientMap gradient_map;
	backward(outputs, 1, tape, gradient_map);
	std::vector<xt::xarray<float>> gradient;
	gradient.reserve(gradient_map.size());
	for (auto i = gradient_map.begin(); i != gradient_map.end(); i++)
		gradient.push_back(i->second);
	return xt::adapt(gradient);
}

void nn::ModelBase::backward(xt::xarray<float>& outputs, xt::xarray<float> deltas, Tape& tape, GradientMap& gradient_map) const
{
	for (auto layer_it = layers.rbegin(); layer_it != layers.rend(); layer_it++)
		(*layer_it)->backward(outputs, deltas, tape, gradient_map);
}

nn::TrainableVars nn::ModelBase::get_trainable_vars() const
{
	TrainableVarsMap trainable_vars_map;
	for (auto& layer : layers)
		layer->get_trainable_vars(trainable_vars_map);
	std::vector<xt::xarray<float>*> trainable_vars;
	for (auto i = trainable_vars_map.begin(); i != trainable_vars_map.end(); i++)
		trainable_vars.push_back(i->second);
	return trainable_vars;
}

nn::TrainableVars nn::ModelBase::get_trainable_vars_fixed() const
{
	TrainableVars trainable_vars;
	for (auto& layer : layers)
		layer->get_trainable_vars(trainable_vars);
	return trainable_vars;
}

void nn::ModelBase::save_weights(const std::string filename) const
{
	const std::vector<xt::xarray<float>*> trainable_vars = get_trainable_vars_fixed();
	std::vector<xt::xarray<float>> weights;
	for (auto& vars : trainable_vars)
		weights.push_back(*vars);
	const nlohmann::json json_weights = weights;
	std::ofstream out_file(filename);
	out_file << json_weights.dump();
	out_file.close();
}

void nn::ModelBase::load_weights(const std::string filename) const
{
	std::ifstream in_file(filename);
	nlohmann::json json_weights;
	in_file >> json_weights;
	in_file.close();
	const std::vector<xt::xarray<float>*> trainable_vars = get_trainable_vars_fixed();
	std::size_t weights_index = 0;
	for (auto& w : json_weights)
	{
		xt::from_json(w, *trainable_vars[weights_index]);
		weights_index++;
	}
}

void nn::ModelBase::print_trainable_vars() const
{
	for (const auto& layer : layers)
		layer->print_trainable_vars();
}