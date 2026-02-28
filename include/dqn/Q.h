#ifndef DQN_Q_H
#define DQN_Q_H

#include <vector>
#include <string>
#include <memory>

namespace dqn
{
	//parameters to configure dqn
	struct QParameters
	{
		float alpha = 0.00025f;
		float gamma = 0.95f;
		float min_eps = 0.1f;
		float max_eps = 0.95f;
		float eps_decr = 0.0001f;
		float beta_min = 0.4f;
		float beta_incr = 0.00005f;
		float priority_scale = 0.6f;
		float min_priority = 0.1f;
		int update_target = 300;
		int train_local = 10;
		std::size_t batch_size = 10;
		std::size_t min_trace = 15;
		std::size_t max_trace = 500;
	};

	class Q final
	{
	private:
		class QPrivate;
		std::unique_ptr<QPrivate> QP;

		//limits for randomizing action number
		static const std::size_t lower_debug = 20;
		static const std::size_t upper_debug = 45;

	public:
		Q(std::size_t field_height, std::size_t field_width, std::size_t channels_number, const std::string player_id, 
			const std::string filepath, const QParameters& custom_parameters = QParameters());
		~Q();
		void soft_reset();
		int call_network(float prev_reward, const std::vector<float>& state,
			const std::vector<float>& actions, std::size_t actions_number);

		int call_network_debug(float prev_reward, std::size_t actions_number);
		int call_network_debug(float prev_reward);
	};
}

#endif