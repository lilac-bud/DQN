#ifndef NEURALNETWORK_LAYER_H
#define NEURALNETWORK_LAYER_H

#include <vector>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xvectorize.hpp>
#include "neural_network/utils/TapeFwd.h"
#include "neural_network/utils/GradientMapFwd.h"
#include "neural_network/utils/TrainableVarsMapFwd.h"

namespace nn
{
	using TrainableVars = std::vector<xt::xarray<float>*>;
	using Axis = int;

	enum class TrainableVarsType
	{
		Weights,
		Biases
	};

	class Layer
	{
	protected:
		static const Axis batch_size_axis = 0;

		static constexpr float lower_rand_bound = 0.0f;
		static constexpr float upper_rand_bound = 0.1f;

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

		virtual void build(std::vector<std::size_t>& shape) = 0;
		virtual void backward(Tape& tape, GradientMap& gradient_map, xt::xarray<float>& deltas) const = 0;
		virtual void get_trainable_vars(TrainableVars& trainable_vars) = 0;
		virtual void get_trainable_vars(TrainableVarsMap& trainable_vars_map) = 0;
		virtual void print_trainable_vars() const = 0;

		void forward(xt::xarray<float>& inputs, Tape* tape) const;

	private:
		virtual void forward(xt::xarray<float>& inputs) const = 0;
	};
}

#endif