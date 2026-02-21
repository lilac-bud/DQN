#ifndef NEURALNETWORK_LAYERFLATTEN_H
#define NEURALNETWORK_LAYERFLATTEN_H

#include "neural_network/layers/Layer.h"

namespace nn
{
	class LayerFlatten : public Layer
	{
	private:
		std::size_t outputs_number = 0;

	public:
		virtual void build(std::vector<std::size_t>& input_shape) override;
		virtual void backward(xt::xarray<float>& outputs, xt::xarray<float>& deltas, Tape& tape, GradientMap& gradient_map) const override;
		virtual void get_trainable_vars(TrainableVars& trainable_vars) override {};
		virtual void get_trainable_vars(TrainableVarsMap& trainable_vars_map) override {};
		virtual void print_trainable_vars() const override {};

	private:
		virtual void forward(xt::xarray<float>& inputs) const override;
	};
}

#endif