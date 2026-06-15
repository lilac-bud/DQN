add_executable(TestQCall)

target_sources(TestQCall
	PRIVATE
		test_q_call.cpp
)

target_link_libraries(TestQCall 
	PRIVATE
		Q
)