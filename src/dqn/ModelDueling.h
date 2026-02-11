#ifndef DQN_MODELDUELING_H
#define DQN_MODELDUELING_H

#include "neural_network/Model.h"

namespace nn
{
	class Layer;
}

namespace dqn
{
	class ModelDueling : public nn::Model
	{
	private:
		std::vector<nn::Layer*> conv_state_branch;
		std::vector<nn::Layer*> conv_actions_branch;
		std::vector<nn::Layer*> value_branch;
		std::vector<nn::Layer*> advantage_branch;

		void call_branch(const std::vector<nn::Layer*>& branch, xt::xarray<float>& inputs,
			std::unordered_map<const nn::Layer*, xt::xarray<float>>* tape) const;
		void get_gradient_from_branch(const std::vector<nn::Layer*>& branch, std::unordered_map<const nn::Layer*, xt::xarray<float>>& tape,
			std::vector<xt::xarray<float>>& gradient, xt::xarray<float>& deltas) const;

		xt::xarray<float> call_with_tape(xt::xarray<float>& state, xt::xarray<float>& actions,
			std::unordered_map<const nn::Layer*, xt::xarray<float>>* tape) const override;

	public:
		ModelDueling();
		void build(std::vector<std::size_t> input_shape) const override;
		xt::xarray<xt::xarray<float>> get_gradient(std::unordered_map<const nn::Layer*,
			xt::xarray<float>>& tape, xt::xarray<float> deltas) const override;
	};
}

#endif