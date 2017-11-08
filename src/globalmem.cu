// C++ 17 Includes:
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <array>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

// Project Includes:
#include "globalmem.cuh"

// Defines:
#define TO_STR(EXPR) ((std::stringstream{} << EXPR).str())
#define NUMELS (1000)

__device__ int d_pGlobalInts[NUMELS];

/**
 * Decrements all values in d_pGlobalInts by 1.0f.
 */
__global__ void decrement_static_kernel()
{
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx > NUMELS)
    return;

  d_pGlobalInts[idx] -= 1;
}

/**
 * Decrements all values in the supplied array by 1.0f.
 */
__global__ void decrement_dynamic_kernel(int* pInts, size_t numInts)
{
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx > numInts)
    return;

  pInts[idx] -= 1;
}

void demoStaticGlobal()
{
  std::array<int, NUMELS> h_ints;
  int cur{0};
  for (int& i : h_ints)
    i = (cur += 1);

  cudaError_t status{cudaSuccess};
  if (cudaSuccess != (status = cudaMemcpyToSymbol(d_pGlobalInts, h_ints.data(), h_ints.size() * sizeof(int))))
    throw std::runtime_error{TO_STR("Global static demo:  Failed to copy memory from host to device: " << status)};

  decrement_static_kernel<<<NUMELS/1024 + 1, 1024>>>();
  if (cudaSuccess != (status = cudaDeviceSynchronize()))
    throw std::runtime_error{TO_STR("Global static demo:  Failed to copy memory from device to host." << status)};

  if (cudaSuccess != (status = cudaMemcpyFromSymbol(h_ints.data(), d_pGlobalInts, h_ints.size() * sizeof(int))))
    throw std::runtime_error{TO_STR("Global static demo:  Failed to copy memory from device to host." << status)};

  for (int i{0}; i < h_ints.size(); ++i)
  {
    if (h_ints[i] != i)
    {
      std::stringstream ss;
      ss << "Unexpected value in static global array at index " << i << ":  " << h_ints[i];
      throw std::runtime_error{ss.str()};
    }
  }
}

void demoDynamicGlobal()
{
  std::vector<int> h_ints;
  std::srand(std::time(0));
  h_ints.resize(std::rand() % 5000); // Don't allocate more than 5000 ints
  std::cout << "Allocated " << h_ints.size() << " ints for dynamic global memory demo.\n";

  for (int i{0}; i < h_ints.size(); ++i)
    h_ints[i] = i + 1;

  int* d_pInts{nullptr};
  cudaError_t status{cudaSuccess};

  if (cudaSuccess != (status = cudaMalloc(&d_pInts, h_ints.size() * sizeof(int))))
    throw std::runtime_error{TO_STR("Global dynamic demo:  Failed to allocate " << h_ints.size() * sizeof(int) << " bytes on the device.")};

  if (cudaSuccess != (status = cudaMemcpy(d_pInts, h_ints.data(), h_ints.size() * sizeof(int), cudaMemcpyHostToDevice)))
  {
    cudaFree(d_pInts);
    throw std::runtime_error{TO_STR("Global dynamic demo:  Failed to copy memory from host to device: " << status)};
  }

  decrement_dynamic_kernel<<<h_ints.size()/1024 + 1, 1024>>>(d_pInts, h_ints.size());
  if (cudaSuccess != (status = cudaDeviceSynchronize()))
  {
    cudaFree(d_pInts);
    throw std::runtime_error{TO_STR("Global dynamic demo:  Failed to copy memory from device to host." << status)};
  }

  if (cudaSuccess != (status = cudaMemcpy(h_ints.data(), d_pInts, h_ints.size() * sizeof(int), cudaMemcpyDeviceToHost)))
  {
    cudaFree(d_pInts);
    throw std::runtime_error{TO_STR("Global dynamic demo:  Failed to copy memory from device to host." << status)};
  }

  if (cudaSuccess != (status = cudaFree(d_pInts)))
  {
    cudaFree(d_pInts);
    throw std::runtime_error{TO_STR("Global dynamic demo:  Failed to free " << h_ints.size() * sizeof(int) << " bytes on the device.")};
  }

  for (int i{0}; i < h_ints.size(); ++i)
  {
    if (h_ints[i] != i)
    {
      std::stringstream ss;
      ss << "Unexpected value in dynamic global array at index " << i << ":  " << h_ints[i];
      throw std::runtime_error{ss.str()};
    }
  }
}
