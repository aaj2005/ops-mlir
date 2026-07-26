# Add Runtime library

find_package(Python3 COMPONENTS Interpreter Development.Embed REQUIRED)

add_library(
	OPSRuntime
	IRBuilder.cpp
	JITEngine.cpp
)

target_include_directories(
	OPSRuntime
	PUBLIC
	${CMAKE_SOURCE_DIR}/include
	${CMAKE_CURRENT_BINARY_DIR}/../Dialect/OPS
)

target_compile_definitions(
	OPSRuntime
	PRIVATE
	OPS_XDSL_DIR="${CMAKE_SOURCE_DIR}/xdsl_impl"
	$<$<BOOL:${OPS_ENABLE_CUDA}>:OPS_ENABLE_CUDA>
)

if(OPS_ENABLE_CUDA)
  target_link_libraries(OPSRuntime PRIVATE CUDA::cuda_driver)
endif()

target_link_libraries(
	OPSRuntime
	PUBLIC
	MLIRIR
	MLIRDialectUtils
	MLIRSupport
	MLIRPass
	MLIROPSDialect
	Python3::Python
)
