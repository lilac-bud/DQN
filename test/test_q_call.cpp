#include "dqn/Q.h"

#include <iostream>
#include <chrono>

int main()
{
	dqn::Q network(10, 10, 2, "0", "");
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 100; ++i)
	{
		std::cout << network.call_network_debug(0) << std::endl;
		if ((i + 1) % 10 == 0)
			network.call_network_debug(1, 0);
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << std::endl << duration.count() << std::endl;
	return 0;
}