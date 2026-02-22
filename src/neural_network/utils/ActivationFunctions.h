#ifndef NEURALNETWORK_ACTIVATIONFUNCTIONS_H
#define NEURALNETWORK_ACTIVATIONFUNCTIONS_H

#include <xtensor/core/xvectorize.hpp>
#include <cmath>
#include <unordered_map>

namespace nn
{
	enum class Activation
	{
		Identity,
		Sigmoid,
		Tanh
	};

	inline float identity_scalar(float input) { return input; }
	inline float identity_derivative_scalar(float input) { return 1; }
	inline float sigmoid_scalar(float input) { return 1 / (1 + std::exp(-input)); }
	inline float sigmoid_derivative_scalar(float input) { return input * (1 - input); }
	inline float tanh_scalar(float input) { return std::tanh(input); }
	inline float tanh_derivative_scalar(float input) { return 1 - (float)pow(input, 2); }

	inline const auto identity = xt::vectorize(identity_scalar);
	inline const auto identity_derivative = xt::vectorize(identity_derivative_scalar);
	inline const auto sigmoid = xt::vectorize(sigmoid_scalar);
	inline const auto sigmoid_derivative = xt::vectorize(sigmoid_derivative_scalar);
	inline const auto tanh = xt::vectorize(tanh_scalar);
	inline const auto tanh_derivative = xt::vectorize(tanh_derivative_scalar);

	template<class E>
	auto activate(const xt::xexpression<E>& input, Activation activation)
	{
		switch (activation)
		{
		case Activation::Sigmoid:
			return sigmoid(input.derived_cast());
		case Activation::Tanh:
			return tanh(input.derived_cast());
		default:
			return identity(input.derived_cast());
		}
	}
	template<class E>
	auto derive(const xt::xexpression<E>& input, Activation activation)
	{
		switch (activation)
		{
		case Activation::Sigmoid:
			return sigmoid_derivative(input.derived_cast());
		case Activation::Tanh:
			return tanh_derivative(input.derived_cast());
		default:
			return identity_derivative(input.derived_cast());
		}
	}
}

#endif