#ifndef NEURALNETWORK_MODELBASE_H
#define NEURALNETWORK_MODELBASE_H

#include "neural_network/utils/TapeFwd.h"

#include <vector>
#include <string>
#include <memory>
#include <xtensor/containers/xarray.hpp>

namespace nn
{
	using TrainableVars = std::vector<xt::xarray<float>*>;

	class Layer;

	class ModelBase
	{
	protected:
		std::vector<std::unique_ptr<Layer>> layers;

		template<typename... Args>
		auto insert_into_layers(Args&&... args)
		{
			std::size_t prev_size = layers.size();
			layers.reserve(prev_size + sizeof...(Args));
			(layers.emplace_back(std::forward<Args>(args)), ...);
			return layers.begin() + prev_size;
		}

	public:
		~ModelBase();

		virtual void build(std::vector<std::size_t> input_shape) const = 0;
		virtual xt::xarray<xt::xarray<float>> get_gradient(Tape& tape, xt::xarray<float> deltas) const = 0;

		TrainableVars get_trainable_vars() const;
		TrainableVars get_trainable_vars_fixed() const;
		void save_weights(const std::string filename) const;
		void load_weights(const std::string filename) const;
		void print_trainable_vars() const;
	};
}

#endif