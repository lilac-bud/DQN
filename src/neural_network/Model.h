#ifndef NEURALNETWORK_MODEL_H
#define NEURALNETWORK_MODEL_H

#include <vector>
#include <map>
#include <string>
#include <xtensor/containers/xarray.hpp>

namespace nn
{
	class Model
	{
	protected:
		std::vector<class Layer*> layers;

		virtual xt::xarray<float> call_with_tape(xt::xarray<float>& state, xt::xarray<float>& actions,
			std::map<const Layer*, xt::xarray<float>>* tape) const = 0;

	public:
		~Model();

		virtual void build(std::vector<std::size_t> input_shape) const = 0;
		virtual xt::xarray<xt::xarray<float>> get_gradient(std::map<const Layer*, 
			xt::xarray<float>>& tape, xt::xarray<float> deltas) const = 0;

		xt::xarray<float> call(xt::xarray<float> state, xt::xarray<float> actions) const;
		xt::xarray<float> call(xt::xarray<float> state, xt::xarray<float> actions, 
			std::map<const Layer*, xt::xarray<float>>* tape) const;
		std::vector<xt::xarray<float>*> get_trainable_vars() const;
		void save_weights(const std::string filename) const;
		void load_weights(const std::string filename) const;
		void print_trainable_vars() const;
	};
}

#endif