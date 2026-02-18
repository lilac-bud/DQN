add_library(neural_network OBJECT)

target_sources(neural_network
	PRIVATE
		neural_network/layers/Layer.cpp
		neural_network/layers/LayerConv2D.cpp
		neural_network/layers/LayerDense.cpp
		neural_network/layers/LayerFlatten.cpp
		neural_network/layers/LayerMaxPooling2D.cpp
		neural_network/model/ModelBase.cpp

	PUBLIC
		FILE_SET HEADERS
		FILES
			neural_network/utils/TapeFwd.h
			neural_network/utils/GradientMapFwd.h
			neural_network/utils/TrainableVarsMapFwd.h
			
			neural_network/layers/Layer.h
			neural_network/layers/LayerConv2D.h
			neural_network/layers/LayerDense.h
			neural_network/layers/LayerFlatten.h
			neural_network/layers/LayerMaxPooling2D.h
			neural_network/model/ModelBase.h
			neural_network/model/ModelCall.h
)

target_link_libraries(neural_network xtensor)