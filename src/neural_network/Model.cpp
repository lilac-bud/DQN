#include "neural_network/Model.h"
#include "neural_network/layers/Layer.h"
#include <xtensor/io/xjson.hpp>
#include <fstream>

nn::Model::~Model()
{
	for (auto& layer : layers)
		delete layer;
}

xt::xarray<float> nn::Model::call(xt::xarray<float> state, xt::xarray<float> actions) const
{
	return call_with_tape(state, actions, nullptr);
}

xt::xarray<float> nn::Model::call(xt::xarray<float> state, xt::xarray<float> actions,
	std::unordered_map<const Layer*, xt::xarray<float>>* tape) const
{
	return call_with_tape(state, actions, tape);
}

std::vector<xt::xarray<float>*> nn::Model::get_trainable_vars() const
{
	std::vector<xt::xarray<float>*> trainable_vars;
	for (auto layer = layers.rbegin(); layer != layers.rend(); layer++)
		(*layer)->get_trainable_vars(trainable_vars);
	return trainable_vars;
}

void nn::Model::save_weights(const std::string filename) const
{
	const std::vector<xt::xarray<float>*> trainable_vars = get_trainable_vars();
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
	const std::vector<xt::xarray<float>*> trainable_vars = get_trainable_vars();
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