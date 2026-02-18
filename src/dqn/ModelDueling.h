#ifndef DQN_MODELDUELING_H
#define DQN_MODELDUELING_H

#include "neural_network/model/ModelBase.h"
#include "neural_network/model/ModelCall.h"
#include "neural_network/utils/GradientMapFwd.h"
#include <array>

namespace dqn
{
	class ModelDueling : public nn::ModelBase, public nn::ModelCall<2>
	{
	private:
		struct layers_part
		{
			std::vector<std::unique_ptr<nn::Layer>>* all_layers;

			std::ptrdiff_t part_begin;
			std::ptrdiff_t part_end;
			std::ptrdiff_t part_rbegin;
			std::ptrdiff_t part_rend;

			auto begin() const { return all_layers->begin() + part_begin; }
			auto end() const { return all_layers->begin() + part_end; }
			auto rbegin() const { return all_layers->rbegin() + part_rbegin; }
			auto rend() const { return all_layers->rbegin() + part_rend; }
		};

		enum LayersPartName
		{
			ConvStatePart, ConvActionsPart, ValuePart, AdvantagePart, 
			PartsTotal,
		};

		enum BranchName
		{
			StateBranch, ActionsBranch, 
			BranchesTotal
		};

		enum GreatPartName
		{
			ConvGreatPart, FlatGreatPart, 
			GreatPartsTotal
		};

		std::array<layers_part, PartsTotal> layers_parts;

		std::array<std::array<LayersPartName, BranchesTotal>, GreatPartsTotal> parts_names = {
			std::array{ConvStatePart, ConvActionsPart},
			std::array{ValuePart, AdvantagePart}
		};

		void call_layers_part(LayersPartName layers_part_name, xt::xarray<float>& inputs, nn::Tape* tape) const;
		void get_gradient_from_layers_part(LayersPartName layers_part_name, nn::Tape& tape, nn::GradientMap& gradient_map,
			xt::xarray<float>& deltas) const;

		virtual xt::xarray<float> call_with_tape(std::array<xt::xarray<float>, 2>& inputs, nn::Tape* tape) const override;

	public:
		ModelDueling();
		virtual void build(std::vector<std::size_t> input_shape) const override;
		virtual xt::xarray<xt::xarray<float>> get_gradient(nn::Tape& tape, xt::xarray<float> deltas) const override;
	};
}

#endif