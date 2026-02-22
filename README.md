# Deep Q-network

Целью разработки этого проекта является решение проблем, возникнувших в процессе разработки проекта для магистерской диссертации. В частности, упрощение установки программы на другие компьютеры. Для этого модуль, написанный на языке Python с использованием tensorflow, был переписан в виде представленной библиотеки.

Библиотека написана на C++ при помощи библиотек [xtensor](https://github.com/xtensor-stack/xtensor), [xtl](https://github.com/xtensor-stack/xtl) (необходима для работы с xtensor) и [nlohmann/json](https://github.com/nlohmann/json) (для сохранения весов и других параметров) и является реализацией [алгоритма DQN](https://github.com/lilac-bud/TBSUE4?tab=readme-ov-file#learning). Для ускорения вычислений используются потоки.

Теоретически библиотеку можно использовать в различных проектах, однако по большей части она предназначена для разработки игр. Пример использования библиотеки в проекте Unreal Engine можно посмотреть [здесь](https://github.com/lilac-bud/TurnBasedStrategy-DQN-CPP).

<!-- TOC-->
  - [1. Установка](#installation)
  - [2. Использование](#use)
  - [3. Внутреннее устройство](#internal_working)
    - [3.1 neural_network](#neural_network)
      - [3.1.1 utils](#utils)
      - [3.1.2 layers](#layers)
      - [3.1.3 model](#model)
    - [3.2 dqn](#dqn)
      - [3.2.1 ModelDueling](#model_dueling)
      - [3.2.2 Q](#q)
<!-- TOC -->

<a name="installation"></a>
## 1. Установка

```bash
cmake --preset dqn
cmake --build build
cmake --install build --prefix <your_install_prefix>
```

<a name="use"></a>
## 2. Использование

Библиотека имеет один публичный хедер Q.h, код которого приведён ниже.
```C++
#ifndef DQN_Q_H
#define DQN_Q_H

#include <vector>
#include <string>
#include <memory>

namespace dqn
{
	class Q final
	{
	public:
		//parameters to configure dqn
		static constexpr float alpha = 0.00025f;
		static constexpr float gamma = 0.95f;
		static constexpr float min_eps = 0.1f;
		static constexpr float max_eps = 0.95f;
		static constexpr float eps_decr = 0.0001f;
		static constexpr float beta_min = 0.4f;
		static constexpr float beta_incr = 0.00005f;
		static constexpr float priority_scale = 0.6f;
		static constexpr float min_priority = 0.1f;
		static const int update_target = 300;
		static const int train_local = 10;
		static const std::size_t batch_size = 10;
		static const std::size_t min_trace = 15;
		static const std::size_t max_trace = 500;

	private:
		class QPrivate;
		std::unique_ptr<QPrivate> QP;

		//limits for randomizing action number
		static const std::size_t lower_debug = 20;
		static const std::size_t upper_debug = 45;

	public:
		Q(std::size_t field_height, std::size_t field_width, std::size_t channels_number,
			const std::string player_id, const std::string filepath);
		~Q();
		void soft_reset();
		int call_network(float prev_reward, const std::vector<float>& state,
			const std::vector<float>& actions, std::size_t actions_number);

		int call_network_debug(float prev_reward, std::size_t actions_number);
		int call_network_debug(float prev_reward);
	};
}

#endif
```
В классе используются следующие константы, которые можно изменять для настройки обучения:
* alpha – скорость обучения;
* gamma – параметр уценки;
* min_eps – минимальное значение ϵ;
* max_eps – максимальное значение ϵ;
* eps_decr – скорость уменьшения ϵ;
* beta_min – минимальное значение β;
* beta_incr – скорость увеличения β;
* priority_scale – степень использования приоритетности;
* min_priority – минимальная приоритетность;
* update_target – периодичность обновления целевой модели;
* train_local – периодичность запуска обучения модели на мини-батче;
* batch_size – размер мини-батча;
* min_trace – минимальный размер истории переходов;
* max_trace – максимальный размер истории переходов. Этот параметр значительно влияет на количество используемой памяти.

Для вызова конструктора класса Q необходимо указать размеры игрового поля и число каналов, которое соответствует количеству типов информации о ячейке игрового поля (например, находится ли в ней вражеский юнит). Также нужно указать ID игрока, для которого Q будет выбирать действие, и путь к папке для сохранений.

Метод soft_reset вызывается по окончании игры (эпизода обучения) для того, чтобы модель могла быть использована для следующей игры.

Метод call_network принимает на вход предыдущую награду, вектор, являющийся представлением текущего состояния (должен быть размера field_height $\times$ field_width $\times$ channels_number), ещё один вектор, соответствующий возможным действиям и число действий (то есть размер вектора actions должен быть field_height $\times$ field_width $\times$ channels_number $\times$ actions_number). Возвращает индекс выбранного действия, если количество переданных действий больше нуля.

Методы call_network_debug используются для отладки и генерируют входные данные в зависимости от параметров.

Загрузка и сохранение не вызываются напрямую.

<a name="internal_working"></a>
## 3. Внутреннее устройство

Для лучшего понимания рекомендуется ознакомиться с [документацией](https://xtensor.readthedocs.io/en/latest/) библиотеки xtensor.

Исходные файлы поделены на две части: 
* neural_network – реализует необходимые элементы для написания нейронных сетей;
* dqn – непосредственно сам алгоритм DQN.

<a name="neural_network"></a>
### 3.1 neural_network

Включает в себя несколько видов слоёв и базовую модель, а также несколько вспомогательных хедеров. 

<a name="utils"></a>
#### 3.1.1 utils

Включает в себя ActivationFunctions.h, где определяются функции активации и их производные. Для этого сначала должны быть написаны скалярные функции, например:
```C++
inline float sigmoid_scalar(float input) { return 1 / (1 + std::exp(-input)); }
inline float sigmoid_derivative_scalar(float input) { return input * (1 - input); }
```
Которые затем векторизуются, чтобы их можно было применять для тензоров:
```C++
inline const auto sigmoid = xt::vectorize(sigmoid_scalar);
inline const auto sigmoid_derivative = xt::vectorize(sigmoid_derivative_scalar);
```
Там же определён enum class Activation со всеми видами функций активаций, что есть в программе, а также шаблонные функции activate и derive, которые принимают на вход выражение [xt::xexpression](https://xtensor.readthedocs.io/en/latest/api/xexpression.html) и вид активации. Именно эти шаблонные функции используются в слоях.

Также utils включает в себя предварительное объявление (forward declaration) следующих псевдонимов типов.
```C++
//TapeFwd.h
using Tape = std::unordered_map<const Layer*, xt::xarray<float>>;
```
Tape применяется для запоминания входных данных всех слоёв модели, которые затем используются при подсчёте производных. Порядок при этом не важен, поэтому используется unordered_map, где ключом является указатель на слой.
```C++
//TrainableVarsMapFwd.h
using TrainableVarsMap = std::map<std::pair<const Layer*, TrainableVarsType>, xt::xarray<float>*>;
//GradientMapFwd.h
using GradientMap = std::map<std::pair<const Layer*, TrainableVarsType>, xt::xarray<float>>;
```
В процессе обучения у всех слоёв модели собираются обучаемые параметры. Для них подсчитываются производные, с которыми затем могут совершаться другие математические манипуляции. Конечный результат складывается с обучаемыми параметрами. 

Очевидно, что полученные производные должны соответствовать обучаемым параметрам, то есть быть в том же порядке. 

При этом у каждого слоя как минимум два вида этих параметров: веса и смещения, – поэтому для сортировки в map используется композитный ключ из указателя на слой и тип параметра, который определён как enum class TrainableVarsType.

Стоит заметить, что так как частью ключа является указатель, при каждом запуске программы порядок будет другой. В случае обучения это не имеет значения. Главное, чтобы порядок производных был такой же, как у обучаемых параметров.

Наконец, в ConvoluteFunction.h определена операция свёртки. Она принимает на вход выражение, которое нужно свернуть, фильтры и размерность выхода.

<a name="layers"></a>
#### 3.1.2 layers

Включает в себя базовый класс nn::Layer:
```C++
//Layer.h
#ifndef NEURALNETWORK_LAYER_H
#define NEURALNETWORK_LAYER_H

#include "neural_network/utils/TapeFwd.h"
#include "neural_network/utils/GradientMapFwd.h"
#include "neural_network/utils/TrainableVarsMapFwd.h"

#include <vector>
#include <xtensor/containers/xarray.hpp>

namespace nn
{
	using TrainableVars = std::vector<xt::xarray<float>*>;
	using Axis = int;

	enum class TrainableVarsType
	{
		Weights,
		Biases
	};

	class Layer
	{
	protected:
		static const Axis batch_size_axis = 0;

		//for randomizing weights and biases
		static constexpr float lower_rand_bound = 0.0f;
		static constexpr float upper_rand_bound = 0.1f;

	public:
		virtual void build(std::vector<std::size_t>& shape) = 0;
		virtual void backward(xt::xarray<float>& outputs, xt::xarray<float>& deltas, Tape& tape, GradientMap& gradient_map) const = 0;
		virtual void get_trainable_vars(TrainableVars& trainable_vars) = 0;
		virtual void get_trainable_vars(TrainableVarsMap& trainable_vars_map) = 0;
		virtual void print_trainable_vars() const = 0;

		void forward(xt::xarray<float>& inputs, Tape* tape) const;

	private:
		virtual void forward(xt::xarray<float>& inputs) const = 0;
	};
}

#endif

//Layer.cpp
#include "neural_network/layers/Layer.h"

void nn::Layer::forward(xt::xarray<float>& inputs, Tape* tape) const
{
	if (tape)
		tape->insert({ this, inputs });
	forward(inputs);
}
```
От этого класса наследуют все остальные слои:
* LayerConv2D – слой свёртки;
* LayerMaxPooling2D – слой подвыборки (субдискретизации);
* LayerFlatten – слой, уменьшающий размерность входа;
* LayerDense – полносвязный слой.

В наследующих слоях должны быть переопределены следующие функции.

virtual void build(std::vector<std::size_t>& shape) получает на вход размерность ожидаемых входных данных и в зависимости от этого может инициализировать внутренние параметры слоёв (к примеру, веса или фильтры). Так же функция изменяет размерность таким образом, чтобы та соответствовала выходным данным слоя (в терминах нейронной сети).

virtual void get_trainable_vars(TrainableVars& trainable_vars) и virtual void get_trainable_vars(TrainableVarsMap& trainable_vars_map) оба нужны для сбора обучаемых параметров слоя (если они у него есть). Разница между ними заключается в используемом контейнере. TrainableVarsMap, как было сказано ранее, является map с композитным ключом, в то время как TrainableVars это просто вектор. Соответственно в одном случае обучаемые параметры будут отсортированы, а в другом будут располагаться в том порядке, в которым были добавлены.

virtual void forward(xt::xarray<float>& inputs) должен осуществлять прямой проход по нейронной сети.

virtual void backward(xt::xarray<float>& outputs, xt::xarray<float>& deltas, Tape& tape, GradientMap& gradient_map), соответственно, обратный. Производные по обучаемым параметрам добавляются в gradient_map. Дельты подсчитываются в соответствии с методом обратного распространения ошибки. Под выходом (outputs) подразумеваются входные данные следующего слоя, которые были сохранены в tape при прямом проходе.

Наконец, virtual void print_trainable_vars() const выводит на экран обучаемые параметры слоя при их наличии.

<a name="model"></a>
#### 3.1.3 model

Базовая модель нейронной сети реализована в виде абстрактных классов ModelBase и ModelCall.

Класс ModelBase включает в себя все необходимые для работы модели методы, кроме вызова. В нём же содержатся слои нейронной сети.
```C++
#ifndef NEURALNETWORK_MODELBASE_H
#define NEURALNETWORK_MODELBASE_H

#include "neural_network/utils/TapeFwd.h"
#include "neural_network/utils/GradientMapFwd.h"

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

		TrainableVars get_trainable_vars() const;
		TrainableVars get_trainable_vars_fixed() const;
		void save_weights(const std::string filename) const;
		void load_weights(const std::string filename) const;
		void print_trainable_vars() const;

		xt::xarray<xt::xarray<float>> get_gradient(xt::xarray<float> outputs, Tape& tape) const;

		virtual void build(std::vector<std::size_t> input_shape) const;

	private:
	virtual void backward(xt::xarray<float>& outputs, xt::xarray<float> deltas, Tape& tape, GradientMap& gradient_map) const;
	};
}

#endif 
```
Для хранения слоёв используются уникальные указатели, которые нельзя копировать. Из-за этого нельзя использовать метод insert, чтобы добавить в вектор сразу несколько элементов. Метод insert_into_layers позволяет обойти эту проблему и так же, как insert, возвращает итератор на первый добавленный элемент.

TrainableVars get_trainable_vars() и TrainableVars get_trainable_vars_fixed() оба возвращают все обучаемые параметры модели. Однако в первом случае для сбора используется TrainableVarsMap, что приводит к сортировке данных по указателю на слой. Если необходимо, чтобы обучаемые параметры не сортировались (к примеру, для их сохранения и загрузки), нужно использовать get_trainable_vars_fixed.

void save_weights(const std::string filename) и void load_weights(const std::string filename), как следует из названий, реализуют сохранение и загрузку обучаемых параметров в указанный файл.

void print_trainable_vars() выводит на экран все обучаемые параметры модели.
В наследующих моделях можно переопределить следующие методы.

virtual void build(std::vector<std::size_t> input_shape) получает на вход размерность ожидаемых входных данных и по умолчанию последовательно вызывает аналогичный метод у всех слоёв модели.

virtual void backward (xt::xarray<float>& outputs, xt::xarray<float> deltas, Tape& tape, GradientMap& gradient_map) используется для сбора производных в gradient_map – обратного прохода. По умолчанию для всех слоёв в обратном порядке вызывается backward.

Соответственно публичный метод xt::xarray<xt::xarray<float>> get_gradient(xt::xarray<float> outputs, Tape& tape) вызывает предыдущий и адаптирует его результат к нужному виду. Этот метод принимает на вход результат вызова модели (outputs) и запомненные на этом вызове входные данные всех слоёв (tape).

Сам вызов модели реализован в виде шаблонного интерфейса ModelCall, который позволяет задать количество ожидаемых входных данных.
```C++
#ifndef NEURALNETWORK_MODELCALL_H
#define NEURALNETWORK_MODELCALL_H

#include "neural_network/utils/TapeFwd.h"

#include <array>
#include <xtensor/containers/xarray.hpp>

namespace nn
{
	template <std::size_t inputs_number>
	class ModelCall
	{
	private:
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
```
virtual xt::xarray<float> call_with_tape(std::array<xt::xarray<float>, inputs_number>& inputs, Tape* tape) должен осуществлять прямой проход по модели и вызывать для всех слоёв метод forward.

<a name="dqn"></a>
### 3.2 dqn

Эта часть непосредственно реализует [алгоритм DQN](https://github.com/lilac-bud/TBSUE4?tab=readme-ov-file#learning) и включает в себя два класса.

<a name="model_dueling"></a>
#### 3.2.1 ModelDueling

Модель, используемая в алгоритме, реализует подход [dueling network](https://github.com/lilac-bud/TurnBasedStrategy-DQN?tab=readme-ov-file#dqn) и состоит из нескольких слоёв [свёрточной сети](https://github.com/lilac-bud/TurnBasedStrategy-DQN?tab=readme-ov-file#cnn) и полносвязных слоёв. Из-за этого слои поделены на несколько частей. Модель вызывается с двумя входами: состоянием и действиями.
```C++
#ifndef DQN_MODELDUELING_H
#define DQN_MODELDUELING_H

#include "neural_network/model/ModelBase.h"
#include "neural_network/model/ModelCall.h"

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
		void backward_layers_part(LayersPartName layers_part_name, xt::xarray<float>& outputs, xt::xarray<float>& deltas, 
			nn::Tape& tape, nn::GradientMap& gradient_map) const;

		virtual xt::xarray<float> call_with_tape(std::array<xt::xarray<float>, 2>& inputs, nn::Tape* tape) const override;
		virtual void backward(xt::xarray<float>& outputs, xt::xarray<float> deltas, nn::Tape& tape,
			nn::GradientMap& gradient_map) const override;

	public:
		ModelDueling();
		virtual void build(std::vector<std::size_t> input_shape) const override;
	};
}

#endif
```
Структура layers_part определена таким образом, чтобы по ней можно было итерировать так же, как по общему вектору со всеми слоями.

Перечисления LayersPartName, BranchName, GreatPartName включены для упрощения доступа к частям.

Массив layers_parts содержит все части слоёв, а parts_names их названия.

Методы call_layers_part и backward_layers_part итерируют по части слоёв, имя которой передаётся в качестве аргумента.

Методы call_with_tape, backward и build переопределены в соответствии с внутренней структурой модели и подходом dueling network.
```C++
//ModelDueling.cpp
xt::xarray<float> dqn::ModelDueling::call_with_tape(std::array<xt::xarray<float>, 2>& inputs, nn::Tape* tape) const
{
	for (int great_part = ConvGreatPart; great_part != GreatPartsTotal; great_part++)
	{
		if (great_part == FlatGreatPart)
		{
			//to get advantage function actions need to be concatenated with the state
			auto& [state, actions] = inputs;
			actions = concatenate(xtuple(broadcast(state, actions.shape()), actions), Axis{ 1 });
		}
		auto& great_part_names = parts_names[great_part];
		for (int branch = StateBranch; branch != BranchesTotal; branch++)
			call_layers_part(great_part_names[branch], inputs[branch], tape);
	}
	auto&[value, advantage] = inputs;
	return value + advantage;
}
```
Сами слои заполняются в конструкторе модели.

<a name="q"></a>
#### 3.2.2 Q

Хедер Q.h уже был описан ранее. Как можно заметить, класс Q написан с использованием идиомы Pimpl, в соответствии с которой объявление приватных структур, переменных и методов перенесено в Q.cpp.

Далее будут рассмотрены подробно только методы QPrivate, непосредственно связанные с реализаций алгоритма DQN.

Для получения индекса действия вызывается get_act. Если были записаны предыдущие состояние и выбранное действие, то вызывается update. После чего выбирается действие. С вероятностью eps оно случайно. В ином случае находится действие с самым большим значением, полученным от вызова локальной модели. Так или иначе состояние и выбранное действие записываются, а выбранный индекс возвращается.
```C++
std::size_t dqn::Q::QPrivate::get_act(float prev_reward, const xt::xarray<float>& state, const xt::xarray<float>& actions)
{
	if (!prev_record.empty)
		update(prev_reward, state, actions, false);
	const std::size_t act_index = random_float() < eps ?
		random_number(0, inputs_number(actions)) : find_best(state, actions, model_local).index;
	prev_record = { state, xt::view(actions, xt::range(act_index, act_index + 1)) };
	return act_index;
}
```
В методе update происходит запись в историю переходов и добавляется новая приоритетность (см. [здесь](https://github.com/lilac-bud/TurnBasedStrategy-DQN?tab=readme-ov-file#priority) про приоритетный выбор переходов). Здесь же вызываются train_model и global_update и обновляются eps и beta. В методе global_update обновляются обучаемые параметры целевой модели.
```C++
void dqn::Q::QPrivate::update(float reward, const xt::xarray<float>& afterstate, const xt::xarray<float>& possible_actions, bool done)
{
	trace.emplace_back( 
		prev_record.state, 
		prev_record.action, 
		reward, 
		afterstate, 
		possible_actions, 
		done );
	add_new_priority();
	if (trace.size() < Q::min_trace)
		return;
	else if (trace.size() > Q::max_trace)
	{
		trace.erase(trace.begin());
		priorities.erase(priorities.begin());
	}
	if (train_count < Q::train_local)
	{
		train_count++;
		return;
	}
	train_model();
	train_count = 0;
	if (eps > Q::min_eps)
		eps -= Q::eps_decr;
	if (beta < 1)
		beta += Q::beta_incr;
	if (update_count < Q::update_target)
	{
		update_count++;
		return;
	}
	global_update();
}
```
В методе train_model выбирается мини-батч переходов (включая веса для [корректировки смещения](https://github.com/lilac-bud/TurnBasedStrategy-DQN?tab=readme-ov-file#priority)). После чего по этому мини-батчу аккумулируются изменения для обучаемых параметров локальной модели, которые затем прибавляются к ним.
```C++
void dqn::Q::QPrivate::train_model()
{
	const auto [index_batch, importance_weights] = get_batch();
	const nn::TrainableVars trainable_vars = model_local.get_trainable_vars();
	xt::xarray<xt::xarray<float>> vars_change;
#if USE_MULTITHREADING_IN_Q
	std::vector<std::thread> threads;
#endif
	for (std::size_t i = 0; i < Q::batch_size; i++)
#if !USE_MULTITHREADING_IN_Q
		accumulate_vars_change(vars_change, index_batch(i), importance_weights(i));
#else
		threads.push_back(std::thread(&QPrivate::accumulate_vars_change, this, 
			std::ref(vars_change), index_batch(i), importance_weights(i)));
	for (auto& thread : threads)
		thread.join();
#endif
	for (std::size_t k = 0; k < trainable_vars.size(); k++)
		*trainable_vars[k] += vars_change(k);
}
```
Изменения обучаемых параметров подсчитываются в accumulate_vars_change. Для сохраненных состояния и действия локальная модель выдаёт Q-значение, по которому затем находятся производные. Затем по формуле, определённой в [алгоритме DQN](https://github.com/lilac-bud/TurnBasedStrategy-DQN?tab=readme-ov-file#dqn), находится целевое значение и ошибка. Изменение обучаемых параметров подсчитывается как производные, помноженные на ошибку, вес для корректировки смещения и скорость обучения. В конце также обновляется приоритетность перехода.
```C++
void dqn::Q::QPrivate::accumulate_vars_change(xt::xarray<xt::xarray<float>>& vars_change, 
	std::size_t trace_index, float importance_weight)
{
	const Transition& transition = trace[trace_index];
	nn::Tape tape;
	const auto l_value = model_local.call({ transition.state, transition.action }, &tape);
	const auto grads = model_local.get_gradient(l_value, tape);
	float target = transition.reward;
	if (transition.done)
		target += Q::gamma * find_best(transition.afterstate, transition.possible_actions, model_target).value;
	const float td_error = target - l_value.TO_SCALAR;
	{
#if USE_MULTITHREADING_IN_Q
		std::lock_guard<std::mutex> lg(vars_change_mutex);
#endif
		vars_change += Q::alpha * importance_weight * td_error * grads;
	}
	//different threads access completely different indeces & no change in the vector's size -> no need to lock
	priorities[trace_index] = std::pow(std::abs(td_error) + Q::min_priority, Q::priority_scale);
}
```
Метод класса Q call_network в зависимости от числа переданных действий вызывает либо get_act, либо update. Как можно понять, get_act практически всегда сам вызывает update, и только в том случае, когда не нужно выбирать индекс действия, update вызывается напрямую.
