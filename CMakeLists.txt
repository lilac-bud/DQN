cmake_minimum_required (VERSION 3.23)

project (dqn)

add_subdirectory(external)
add_subdirectory(src)

add_library(Q STATIC)

target_sources(Q
	PRIVATE
		src/dqn/Q.cpp
		src/dqn/ModelDueling.cpp

	PRIVATE
		FILE_SET privateHeaders
		TYPE HEADERS
		BASE_DIRS
			src
		FILES
			src/dqn/ModelDueling.h
			
	PUBLIC
		FILE_SET publicHeaders
		TYPE HEADERS
		BASE_DIRS
			include
		FILES
			include/dqn/Q.h
)

target_link_libraries(Q
	PRIVATE
		json
		xtensor
		neural_network
)