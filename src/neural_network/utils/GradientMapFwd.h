#ifndef NEURALNETWORK_GRADIENTMAPFWD_H
#define NEURALNETWORK_GRADIENTMAPFWD_H

#include <map>
#include <xtensor/containers/xarray.hpp>

namespace nn
{
	enum class TrainableVarsType;
	class Layer;

	using GradientMap = std::map<std::pair<const Layer*, TrainableVarsType>, xt::xarray<float>>;
}

#endif
