#ifndef NEURALNETWORK_TRAINABLEVARSMAPFWD_H
#define NEURALNETWORK_TRAINABLEVARSMAPFWD_H

#include <map>
#include <xtensor/containers/xarray.hpp>

namespace nn
{
	enum class TrainableVarsType;
	class Layer;
	
	using TrainableVarsMap = std::map<std::pair<const Layer*, TrainableVarsType>, xt::xarray<float>*>;
}

#endif