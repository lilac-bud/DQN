#ifndef NEURALNETWORK_LAYERMAXPOOLING2D_H
#define NEURALNETWORK_LAYERMAXPOOLING2D_H

#include "neural_network/layers/Layer.h"

namespace nn
{
	class LayerMaxPooling2D : public Layer
	{
	protected:
		std::vector<std::size_t> outputs_shape;
		std::vector<std::size_t> pool_size;

	public:
		LayerMaxPooling2D(std::vector<std::size_t> pool_size);
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