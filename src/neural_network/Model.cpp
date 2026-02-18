#include "neural_network/Model.h"
#include "neural_network/layers/Layer.h"
#include <xtensor/io/xjson.hpp>
#include <fstream>

nn::Model::~Model() = default;

xt::xarray<float> nn::Model::call(xt::xarray<float> state, xt::xarray<float> actions) const
{
	return call_with_tape(state, actions, nullptr);
}

xt::xarray<float> nn::Model::call(xt::xarray<float> state, xt::xarray<float> actions, Tape* tape) const
{
	return call_with_tape(state, actions, tape);
}

nn::TrainableVars nn::Model::get_trainable_vars() const
{
	TrainableVarsMap trainable_vars_map;
	for (auto& layer : layers)
		layer->get_trainable_vars(trainable_vars_map);
	std::vector<xt::xarray<float>*> trainable_vars;
	for (auto i = trainable_vars_map.begin(); i != trainable_vars_map.end(); i++)
		trainable_vars.push_back(i->second);
	return trainable_vars;
}

nn::TrainableVars nn::Model::get_trainable_vars_fixed() const
{
	TrainableVars trainable_vars;
	for (auto& layer : layers)
		layer->get_trainable_vars(trainable_vars);
	return trainable_vars;
}

void nn::Model::save_weights(const std::string filename) const
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

void nn::Model::load_weights(const std::string filename) const
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

void nn::Model::print_trainable_vars() const
{
	for (const auto& layer : layers)
		layer->print_trainable_vars();
}