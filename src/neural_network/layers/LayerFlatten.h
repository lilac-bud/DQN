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
		virtual void backward(std::map<const Layer*, xt::xarray<float>>& tape, std::vector<xt::xarray<float>>& gradient,
			xt::xarray<float>& deltas) const override;
		virtual void get_trainable_vars(std::vector<xt::xarray<float>*>& trainable_vars) override {};
		virtual void print_trainable_vars() const override {};

	private:
		virtual void forward(xt::xarray<float>& inputs) const override;
	};
}

#endif