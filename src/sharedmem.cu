// C++ 17 Includes:
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <array>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <numeric>

// Project Includes:
#include "sharedmem.cuh"

// Defines:
#define TO_STR(EXPR) ((std::stringstream{} << EXPR).str())

__global__ void sum_dynamic_kernel(const int* pIn, int* pOut, size_t numInts)
{
  extern __shared__ int ps[]; // Automatically points to our shared memory array
  
  // Load shared memory:
  ps[threadIdx.x] = pIn[threadIdx.x];
  if (threadIdx.x + blockDim.x <  numInts)
    ps[threadIdx.x + blockDim.x] = pIn[threadIdx.x + blockDim.x];
  if (0 == threadIdx.x && 1 == (1 & numInts))
    ps[numInts - 1] = pIn[numInts - 1];

  size_t prevNumThreads{numInts};
  for (size_t numThreads{blockDim.x}; numThreads > 0; numThreads >>= 1)
  {
    if (threadIdx.x > numThreads)
      return;

    ps[threadIdx.x] += ps[threadIdx.x + numThreads];
    if (1 == (prevNumThreads & 1))
      ps[0] += ps[prevNumThreads - 1];

    prevNumThreads = numThreads;
  }

  *pOut = ps[0];
}

void demoDynamicShared()
{
  std::vector<int> h_ints;
  std::srand(std::time(0));
  h_ints.resize(std::rand() % 1025); // Don't allocate more than we can bite off in one block
  std::cout << "Allocated " << h_ints.size() << " ints for dynamic shared memory demo.\n";
  for (int i{0}; i < h_ints.size(); ++i)
    h_ints[i] = i + 1;

  int* d_pInts{nullptr};
  cudaError_t status{cudaSuccess};

  if (cudaSuccess != (status = cudaMalloc(&d_pInts, (h_ints.size() + 1) * sizeof(int))))
    throw std::runtime_error{TO_STR("Global dynamic demo:  Failed to allocate " << h_ints.size() * sizeof(int) << " bytes on the device.")};

  if (cudaSuccess != (status = cudaMemcpy(d_pInts, h_ints.data(), h_ints.size() * sizeof(int), cudaMemcpyHostToDevice)))
  {
    cudaFree(d_pInts);
    throw std::runtime_error{TO_STR("Global dynamic demo:  Failed to copy memory from host to device: " << status)};
  }

  sum_dynamic_kernel<<<1, h_ints.size() / 2, h_ints.size() * sizeof(int)>>>(d_pInts, d_pInts + h_ints.size(), h_ints.size());
  if (cudaSuccess != (status = cudaDeviceSynchronize()))
  {
    cudaFree(d_pInts);
    throw std::runtime_error{TO_STR("Global dynamic demo:  Failed to copy memory from device to host." << status)};
  }

  int result{0};
  if (cudaSuccess != (status = cudaMemcpy(&result, d_pInts + h_ints.size(), sizeof(int), cudaMemcpyDeviceToHost)))
  {
    cudaFree(d_pInts);
    throw std::runtime_error{TO_STR("Global dynamic demo:  Failed to copy memory from device to host." << status)};
  }

  if (cudaSuccess != (status = cudaFree(d_pInts)))
    throw std::runtime_error{TO_STR("Global dynamic demo:  Failed to free " << h_ints.size() * sizeof(int) << " bytes on the device.")};

  int correctResult{-1};
  if (result != (correctResult = std::accumulate(h_ints.begin(), h_ints.end(), 0)))
    throw std::runtime_error{TO_STR("Shared dynamic demo:  Got the wrong answer (correct: " << correctResult << ", our answer: " << result << ")")};
}
