#include <vector>

#include <jni.h>

#include "VectorAdd.h"

__global__ void vector_add_kernel(int *a, int *b, int *c, size_t len) {

  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;

  if (offset < len) {
    c[offset] = a[offset] + b[offset];
  }

}

JNIEXPORT jintArray JNICALL Java_VectorAdd_add(JNIEnv *env, jclass thisClass,
		                               jintArray a, jintArray b) {

  size_t len = env->GetArrayLength(a);

  int *h_a = env->GetIntArrayElements(a, nullptr);
  int *h_b = env->GetIntArrayElements(b, nullptr);
  std::vector<int> h_c(len);

  int *d_a, *d_b, *d_c;

  cudaMalloc(&d_a, sizeof(int) * len);
  cudaMalloc(&d_b, sizeof(int) * len);
  cudaMalloc(&d_c, sizeof(int) * len);

  cudaMemcpy(d_a, h_a, sizeof(int) * len, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(int) * len, cudaMemcpyHostToDevice);

  const int block_size = 32;
  const int num_blocks = len / block_size + 1;

  vector_add_kernel<<<num_blocks, block_size>>>(d_a, d_b, d_c, len);

  cudaMemcpy(h_c.data(), d_c, sizeof(int) * len, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  env->ReleaseIntArrayElements(a, h_a, 0);
  env->ReleaseIntArrayElements(b, h_b, 0);

  jintArray c = env->NewIntArray(len);

  env->SetIntArrayRegion(c, 0, len, h_c.data());

  return c;
}
