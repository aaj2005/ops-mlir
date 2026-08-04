# Add Runtime library

find_package(Python3 COMPONENTS Interpreter Development.Embed REQUIRED)

add_library(
	OPSRuntime
	IRBuilder.cpp
	JITEngine.cpp
	KernelIRBuilder.cpp
	KernelProfiler.cpp
)

target_include_directories(
	OPSRuntime
	PUBLIC
	${CMAKE_SOURCE_DIR}/include
	${CMAKE_CURRENT_BINARY_DIR}/../Dialect/OPS
	PRIVATE
	${CLANG_INCLUDE_DIRS}
)

# KernelIRBuilder.cpp needs Clang's resource-dir (its own bundled
# stddef.h/stdarg.h/etc, resolved relative to the clang binary) since it
# builds ASTs via the library API directly, bypassing the driver logic
# that normally infers this.
file(GLOB CLANG_RESOURCE_DIR_CANDIDATES "${LLVM_BUILD_DIR}/lib/clang/*")
list(GET CLANG_RESOURCE_DIR_CANDIDATES 0 CLANG_RESOURCE_DIR)

target_compile_definitions(
	OPSRuntime
	PRIVATE
	OPS_XDSL_DIR="${CMAKE_SOURCE_DIR}/xdsl_impl"
	OPS_CLANG_RESOURCE_DIR="${CLANG_RESOURCE_DIR}"
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
	MLIRExecutionEngine
	MLIRTargetLLVMIRExport
	MLIRToLLVMIRTranslationRegistration
	MLIRVectorToLLVM
	MLIRRegisterAllExtensions
	Python3::Python
	clangTooling
)
