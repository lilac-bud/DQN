#include "neural_network/layers/Layer.h"

void nn::Layer::forward(xt::xarray<float>& inputs, Tape* tape) const
{
	if (tape)
		tape->insert({ this, inputs });
	forward(inputs);
}