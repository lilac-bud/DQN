#ifndef NEURALNETWORK_LAYERMAXPOOLING2D_H
#define NEURALNETWORK_LAYERMAXPOOLING2D_H

#include "neural_network/layers/Layer.h"

namespace nn
{
	using PoolSize = std::pair<std::size_t, std::size_t>;

	class LayerMaxPooling2D : public Layer
	{
	protected:
		static const Axis height_axis = 1;
		static const Axis width_axis = 2;
		static const Axis channels_axis = 3;

		std::vector<std::size_t> outputs_shape;
		std::size_t pool_height;
		std::size_t pool_width;

	public:
		LayerMaxPooling2D(PoolSize pool_size);
		virtual void build(std::vector<std::size_t>& input_shape) override;
		virtual void backward(Tape& tape, GradientMap& gradient_map, xt::xarray<float>& deltas) const override;
		virtual void get_trainable_vars(TrainableVars& trainable_vars) override {};
		virtual void get_trainable_vars(TrainableVarsMap& trainable_vars_map) override {};
		virtual void print_trainable_vars() const override {};

	private:
		virtual void forward(xt::xarray<float>& inputs) const override;
	};
}

#endif