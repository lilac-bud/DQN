#include "dqn/Q.h"
#include "dqn/ModelDueling.h"
#include <xtensor/views/xindex_view.hpp>
#include <xtensor/misc/xsort.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <fstream>
#include <ctime>
#define USE_MULTITHREADING_IN_Q 1
#if USE_MULTITHREADING_IN_Q
#include <mutex>
#include <thread>
#endif

struct PreviousStateAction
{
	xt::xarray<float> state;
	xt::xarray<float> action;
	bool empty = true;
};

struct Transition
{
	xt::xarray<float> state;
	xt::xarray<float> action;
	float reward;
	xt::xarray<float> afterstate;
	xt::xarray<float> possible_actions;
	bool done;
};

class dqn::Q::QPrivate final
{
private:
	std::string model_local_filename;
	std::string model_target_filename;
	std::string parameters_filename;
	float eps = Q::max_eps;
	float beta = Q::beta_min;
	int update_count = Q::update_target;
	int train_count = Q::train_local;

	ModelDueling model_local;
	ModelDueling model_target;
	std::vector<Transition> trace;
	std::vector<float> priorities;

#if USE_MULTITHREADING_IN_Q
	std::mutex vars_change_mutex;
#endif

	void load();
	void save() const;
	std::tuple<std::size_t, float> find_best(const xt::xarray<float>& state, const xt::xarray<float>& actions,
		const ModelDueling& model) const;
	void get_batch(xt::xarray<std::size_t>& index_batch, xt::xarray<float>& importance_weights) const;
	void train_model();
	void accumulate_vars_change(xt::xarray<xt::xarray<float>>& vars_change, std::size_t trace_index, float importance_weight);
	void global_update();
	void add_new_priority();

public:
	~QPrivate();

	std::size_t field_height = 0;
	std::size_t field_width = 0;
	std::size_t channels_number = 0;

	PreviousStateAction prev_record;

	QPrivate(std::size_t field_height, std::size_t field_width, std::size_t channels_number,
		const std::string player_id, const std::string filepath);
	std::size_t get_act(float prev_reward, const xt::xarray<float>& state, const xt::xarray<float>& actions);
	void update(float reward, const xt::xarray<float>& afterstate, const xt::xarray<float>& possible_actions, bool done);
};

dqn::Q::Q(std::size_t field_height, std::size_t field_width, std::size_t channels_number,
	const std::string player_id, const std::string filepath)
{
	xt::random::seed(time(nullptr));
	QP = std::make_unique<QPrivate>(field_height, field_width, channels_number, player_id, filepath);
}

dqn::Q::~Q() = default;

dqn::Q::QPrivate::QPrivate(std::size_t field_height, std::size_t field_width, std::size_t channels_number,
	const std::string player_id, const std::string filepath)
{
	this->field_height = field_height;
	this->field_width = field_width;
	this->channels_number = channels_number;
	model_local.build({ field_height, field_width, channels_number });
	model_target.build({ field_height, field_width, channels_number });
	const std::string common_part = filepath + std::string("qdb") + std::to_string(field_height) + std::string("x") +
		std::to_string(field_width) + std::string("_") + player_id;
	model_local_filename = common_part + "_local.json";
	model_target_filename = common_part + "_target.json";
	parameters_filename = common_part + "_parameters.json";
	load();
}

dqn::Q::QPrivate::~QPrivate()
{
	save();
}

void dqn::Q::soft_reset()
{
	QP->prev_record = PreviousStateAction();
}

void dqn::Q::QPrivate::save() const
{
	model_local.save_weights(model_local_filename);
	model_target.save_weights(model_target_filename);
	const nlohmann::json parameters = {
		{"eps", eps},
		{"beta", beta},
		{"update_count", update_count} };
	std::ofstream out_file(parameters_filename);
	out_file << parameters.dump();
	out_file.close();
}

void dqn::Q::QPrivate::load()
{
	if (std::filesystem::exists(model_local_filename) && std::filesystem::exists(model_local_filename) 
		&& std::filesystem::exists(parameters_filename))
	{
		model_local.load_weights(model_local_filename);
		model_target.load_weights(model_target_filename);
		nlohmann::json parameters;
		std::ifstream in_file(parameters_filename);
		in_file >> parameters;
		in_file.close();
		eps = parameters["eps"];
		beta = parameters["beta"];
		update_count = parameters["update_count"];
	}
	else
		global_update();
}

int dqn::Q::call_network(float prev_reward, const std::vector<float>& state, const std::vector<float>& actions, 
	std::size_t actions_number)
{
	const xt::xarray<float> adapted_state = xt::adapt(state, 
		std::vector<std::size_t>({ 1, QP->field_height, QP->field_width, QP->channels_number }));
	xt::xarray<float> adapted_actions;
	if (actions_number > 0)
	{
		adapted_actions = xt::adapt(actions, 
			std::vector<std::size_t>({ actions_number, QP->field_height, QP->field_width, QP->channels_number }));
		return (int)QP->get_act(prev_reward, adapted_state, adapted_actions);
	}
	else
	{
		QP->update(prev_reward, adapted_state, adapted_actions, true);
		return -1;
	}
}

int dqn::Q::call_network_debug(float prev_reward, std::size_t actions_number)
{
	const xt::xarray<float> state = xt::random::rand<float>({ (std::size_t)1, QP->field_height, QP->field_width, QP->channels_number });
	xt::xarray<float> actions;
	if (actions_number > 0)
	{
		actions = xt::random::rand<float>({ actions_number, QP->field_height, QP->field_width, QP->channels_number });
		return (int)QP->get_act(prev_reward, state, actions);
	}
	else
	{
		QP->update(prev_reward, state, actions, true);
		return -1;
	}
}

int dqn::Q::call_network_debug(float prev_reward)
{
	return call_network_debug(prev_reward, 
		xt::random::randint<std::size_t>({ 1 }, lower_actions_number_limit_debug, upper_actions_number_limit_debug)(0));
}

std::tuple<std::size_t, float> dqn::Q::QPrivate::find_best(const xt::xarray<float>& state, const xt::xarray<float>& actions,
	const ModelDueling& model) const
{
	const auto values = model.call(state, actions);
	const std::size_t max_index = xt::unique(values).size() == 1 ?
		xt::random::randint<std::size_t>({ 1 }, 0, actions.shape()[0])(0) : xt::argmax(values)(0);
	return std::make_tuple(max_index, values(max_index));
}

std::size_t dqn::Q::QPrivate::get_act(float prev_reward, const xt::xarray<float>& state, const xt::xarray<float>& actions)
{
	if (!prev_record.empty)
		update(prev_reward, state, actions, false);
	const std::size_t act_index = xt::random::rand<float>({ 1 })(1) < eps ?
		xt::random::randint<std::size_t>({ 1 }, 0, actions.shape()[0])(0) : std::get<0>(find_best(state, actions, model_local));
	prev_record = { state, xt::view(actions, xt::range(act_index, act_index + 1), xt::all()), false };
	return act_index;
}

void dqn::Q::QPrivate::get_batch(xt::xarray<std::size_t>& index_batch, xt::xarray<float>& importance_weights) const
{
	auto adapted_priorities = xt::adapt(priorities);
	index_batch = xt::random::choice(xt::arange(trace.size()), Q::batch_size, adapted_priorities, false);
	auto inter_res = xt::index_view(adapted_priorities, index_batch) / xt::sum(adapted_priorities) * trace.size();
	importance_weights = xt::pow(inter_res, -beta) / xt::amax(inter_res);
}

void dqn::Q::QPrivate::train_model()
{
	xt::xarray<std::size_t> index_batch;
	xt::xarray<float> importance_weights;
	get_batch(index_batch, importance_weights);
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

void dqn::Q::QPrivate::accumulate_vars_change(xt::xarray<xt::xarray<float>>& vars_change, std::size_t trace_index, float importance_weight)
{
	const Transition& transition = trace[trace_index];
	nn::Tape tape;
	const auto l_value = model_local.call(transition.state, transition.action, &tape);
	const auto grads = model_local.get_gradient(tape, l_value);
	const float target = transition.done ? transition.reward :
		transition.reward + Q::gamma * std::get<1>(find_best(transition.afterstate, transition.possible_actions, model_target));
	const float td_error = target - l_value(0);
	{
#if USE_MULTITHREADING_IN_Q
		std::lock_guard<std::mutex> lg(vars_change_mutex);
#endif
		vars_change += Q::alpha * importance_weight * td_error * grads;
	}
	//different threads access completely different indeces & no change in the vector's size -> no need to lock
	priorities[trace_index] = std::pow(std::abs(td_error) + Q::min_priority, Q::priority_scale);
}

void dqn::Q::QPrivate::update(float reward, const xt::xarray<float>& afterstate, const xt::xarray<float>& possible_actions, bool done)
{
	trace.push_back({ prev_record.state, prev_record.action, reward, afterstate, possible_actions, done });
	add_new_priority();
	if (trace.size() < Q::min_trace)
		return;
	else if (trace.size() > Q::max_trace)
	{
		/*float max_p = 0.0f;
		for (float p : priorities)
			if (p > max_p) max_p = p;
		for (auto i = priorities.begin(); i != priorities.end();)
			if (*i < max_p)
			{
				trace.erase(trace.begin() + (i - priorities.begin()));
				i = priorities.erase(i);
			}
			else
				i++;*/
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

void dqn::Q::QPrivate::global_update()
{
	const nn::TrainableVars local_trainable_vars = model_local.get_trainable_vars_ordered();
	const nn::TrainableVars target_trainable_vars = model_target.get_trainable_vars_ordered();
	for (std::size_t i = 0; i < target_trainable_vars.size(); i++)
		*target_trainable_vars[i] = *local_trainable_vars[i];
	update_count = 0;
}

void dqn::Q::QPrivate::add_new_priority()
{
	if (priorities.empty())
	{
		priorities.push_back(1.0f); 
		return;
	}
	float new_p = 0.0f;
	for (float p : priorities)
		if (p > new_p)
			new_p = p;
	priorities.push_back(new_p);
}

#undef USE_MULTITHREADING_IN_Q