set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_DEVICE_LINK_EXECUTABLE ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -m64 -03 -arch=sm_30 -c --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v)

set(CUDA_SOURCES
		"wrappers.cuh"
		"integration_kernel.cuh"
		"solver_kernel.cuh"
		"kernel.cuh"
		"util.cuh"
		"util.cu"
		"integration.cu"
		"solver.cu"
		"shared_variables.cu"
		"shared_variables.cuh"
		"helper_cuda.h"
)

add_library(cuda OBJECT ${CUDA_SOURCES})
target_link_libraries(cuda cudart curand)
