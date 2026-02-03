add_library(json INTERFACE)

target_sources(json
	INTERFACE
		FILE_SET HEADERS
)

add_library(xtl INTERFACE)

target_sources(xtl
	INTERFACE
		FILE_SET HEADERS
)

target_compile_features(xtl INTERFACE cxx_std_17)

add_library(xtensor INTERFACE)

target_sources(xtensor
	INTERFACE
		FILE_SET HEADERS
)

target_compile_features(xtensor INTERFACE cxx_std_20)

target_link_libraries(xtensor INTERFACE xtl)