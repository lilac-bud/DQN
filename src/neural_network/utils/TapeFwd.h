#ifndef NEURALNETWORK_TAPEFWD_H
#define NEURALNETWORK_TAPEFWD_H

#include <unordered_map>
#include <xtensor/containers/xarray.hpp>

namespace nn
{
	class Layer;

	using Tape = std::unordered_map<const Layer*, xt::xarray<float>>;
}

#endif