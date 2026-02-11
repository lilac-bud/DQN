#include "neural_network/layers/Layer.h"

void nn::Layer::forward(xt::xarray<float>& inputs, std::unordered_map<const Layer*, xt::xarray<float>>* tape) const
{
	if (tape)
		tape->insert({ this, inputs });
	forward(inputs);
}