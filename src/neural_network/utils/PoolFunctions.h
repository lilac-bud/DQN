#ifndef NEURALNETWORK_POOLFUNCTIONS_H
#define NEURALNETWORK_POOLFUNCTIONS_H

#include <xtensor/views/xview.hpp>
#include <xtensor/views/xindex_view.hpp>

namespace nn
{
	using Axis = int;
	using PoolSize = std::pair<std::size_t, std::size_t>;

	const Axis height_axis = 1;
	const Axis width_axis = 2;
	const Axis channels_axis = 3;

	template<class I, class S>
	auto maxpool2D(const I& inputs, const S& outputs_shape, PoolSize pool_size)
	{
		auto& [pool_height, pool_width] = pool_size;
		auto outputs = xt::xarray<float>::from_shape(outputs_shape);
		for (std::size_t i = 0; i < outputs_shape[height_axis]; ++i)
		{
			const std::size_t inputs_i = i * pool_height;
			for (std::size_t k = 0; k < outputs_shape[width_axis]; ++k)
			{
				const std::size_t inputs_k = k * pool_width;
				//pool is a part of input taken according to pool size
				auto pool = xt::view(inputs, xt::all(), xt::range(inputs_i, inputs_i + pool_height),
					xt::range(inputs_k, inputs_k + pool_width));
				xt::view(outputs, xt::all(), i, k) = xt::amax(pool, { height_axis, width_axis });
			}
		}
		return outputs;
	}

	template<class O, class D, class N>
	auto unmaxpool2D(N new_deltas, const O& outputs, const D& deltas, PoolSize pool_size)
	{
		auto& [pool_height, pool_width] = pool_size;
		for (std::size_t i = 0; i < outputs.shape()[height_axis]; ++i)
		{
			const std::size_t i_start = i * pool_height;
			for (std::size_t k = 0; k < outputs.shape()[width_axis]; ++k)
			{
				const std::size_t k_start = k * pool_width;
				auto pool = xt::view(new_deltas, xt::all(), xt::range(i_start, i_start + pool_height),
					xt::range(k_start, k_start + pool_width));
				auto max = xt::view(outputs, xt::all(), xt::range(i, i + 1), xt::range(k, k + 1));
				xt::filtration(pool, pool < max) = 0;
				xt::filtration(pool, pool > 0) = 1;
				pool *= xt::view(deltas, xt::all(), xt::range(i, i + 1), xt::range(k, k + 1));
			}
		}
		return new_deltas;
	}
}

#endif