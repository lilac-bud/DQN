#ifndef NEURALNETWORK_MODELCALL_H
#define NEURALNETWORK_MODELCALL_H

#include <array>
#include <vector>
#include <xtensor/containers/xarray.hpp>
#include "neural_network/utils/TapeFwd.h"

namespace nn
{
	template <std::size_t inputs_number>
	class ModelCall
	{
	protected:
		virtual xt::xarray<float> call_with_tape(std::array<xt::xarray<float>, inputs_number>& inputs, Tape* tape) const = 0;

	public:
		xt::xarray<float> call(std::array<xt::xarray<float>, inputs_number> inputs) const
		{
			return call_with_tape(inputs, nullptr);
		}
		xt::xarray<float> call(std::array<xt::xarray<float>, inputs_number> inputs, Tape* tape) const
		{
			return call_with_tape(inputs, tape);
		}
	};
}

#endif
