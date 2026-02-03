#ifndef NEURALNETWORK_LAYER_H
#define NEURALNETWORK_LAYER_H

#include <vector>
#include <map>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xvectorize.hpp>

namespace nn
{
	class Layer
	{
	public:
		static float sigmoid_scalar(const float input)
		{
			return 1 / (1 + input);
		}
		static float sigmoid_derivative_scalar(const float input)
		{
			return input * (1 - input);
		}
		static inline auto sigmoid = xt::vectorize(sigmoid_scalar);
		static inline auto sigmoid_derivative = xt::vectorize(sigmoid_derivative_scalar);

		virtual void build(std::vector<std::size_t>& input_shape) = 0;
		virtual void forward(xt::xarray<float>& inputs, std::map<const Layer*, xt::xarray<float>>* tape) const = 0;
		virtual void backward(std::map<const Layer*, xt::xarray<float>>& tape, std::vector<xt::xarray<float>>& gradient,
			xt::xarray<float>& deltas) const = 0;
		virtual void get_trainable_vars(std::vector<xt::xarray<float>*>& trainable_vars) = 0;
		virtual void print_trainable_vars() const = 0;
	};
}

#endif