#ifndef NEURALNETWORK_LAYERMAXPOOLING2D_H
#define NEURALNETWORK_LAYERMAXPOOLING2D_H

#include "neural_network/layers/Layer.h"

namespace nn
{
	using PoolSize = std::pair<std::size_t, std::size_t>;

	class LayerMaxPooling2D : public Layer
	{
	protected:
		std::vector<std::size_t> outputs_shape;
		PoolSize pool_size;

	public:
		LayerMaxPooling2D(PoolSize pool_size);
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