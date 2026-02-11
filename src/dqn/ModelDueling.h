#ifndef DQN_MODELDUELING_H
#define DQN_MODELDUELING_H

#include "neural_network/Model.h"
#include "neural_network/utils/TapeFwd.h"
#include "neural_network/utils/GradientMapFwd.h"

namespace dqn
{
	class ModelDueling : public nn::Model
	{
	private:
		std::vector<nn::Layer*> conv_state_branch;
		std::vector<nn::Layer*> conv_actions_branch;
		std::vector<nn::Layer*> value_branch;
		std::vector<nn::Layer*> advantage_branch;

		void call_branch(const std::vector<nn::Layer*>& branch, xt::xarray<float>& inputs, nn::Tape* tape) const;
		void get_gradient_from_branch(const std::vector<nn::Layer*>& branch, nn::Tape& tape, nn::GradientMap& gradient_map, 
			xt::xarray<float>& deltas) const;

		xt::xarray<float> call_with_tape(xt::xarray<float>& state, xt::xarray<float>& actions, nn::Tape* tape) const override;

	public:
		ModelDueling();
		void build(std::vector<std::size_t> input_shape) const override;
		xt::xarray<xt::xarray<float>> get_gradient(nn::Tape& tape, xt::xarray<float> deltas) const override;
	};
}

#endif