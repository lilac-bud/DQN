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
		static const std::size_t max_trace = 3000;

	private:
		class QPrivate;
		std::unique_ptr<QPrivate> QP;

		static const std::size_t lower_actions_number_limit_debug = 20;
		static const std::size_t upper_actions_number_limit_debug = 45;

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