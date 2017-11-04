#include <iostream>
#include <stdexcept>
#include <sstream>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "als.h"

namespace implicit {

using std::invalid_argument;

__inline__ __device__ void dot(float * out, const float * a, const float * b) {
    if (threadIdx.x == 0) {
        out[0] = 0;
    }
    __syncthreads();
    // The vast majority of time is spent in the next line
    atomicAdd(out, a[threadIdx.x] * b[threadIdx.x]);
    __syncthreads();
}

__global__ void least_squares_cg_kernel(int factors, int user_count, int item_count, float * X,
                                        const float * Y, const float * YtY,
                                        const int * indptr, const int * indices, const float * data) {
    // use shared memory for temporary variables
    extern __shared__ float shared_memory[];

    // Ap/r/p are vectors for CG update
    float * Ap = &shared_memory[0];
    float * r = &shared_memory[factors];
    float * p = &shared_memory[2*factors];

    // float values that need shared between threads
    float * rsold = &shared_memory[3*factors];
    float * rsnew = &shared_memory[3*factors + 1];
    float * temp = &shared_memory[3*factors + 2];

    // https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int u = blockIdx.x; u < user_count; u += gridDim.x) {
        float * x = &X[u * factors];

        // calculate residual r = YtCuPu - YtCuY Xu = YtCuPu - Yt(Cu-1)yYXu - YtYXu
        float YtYXu = 0;
        for (int i = 0; i < factors; ++i) {
            YtYXu += x[i] * YtY[i * factors + threadIdx.x];
        }

        r[threadIdx.x] = -YtYXu;
        for (int index = indptr[u]; index < indptr[u + 1]; ++index) {
            int i = indices[index];
            float confidence = data[index];

            dot(temp, &Y[i * factors], x);

            r[threadIdx.x] += (confidence - (confidence - 1) * temp[0]) * Y[i * factors + threadIdx.x];
        }
        p[threadIdx.x] = r[threadIdx.x];

        dot(rsold, r, r);

        const int cg_steps = 3;
        for (int it = 0; it < cg_steps; ++it) {
            // calculate Ap = YtCuYp - without actually calculating YtCuY
            Ap[threadIdx.x] = 0;
            for (int i = 0; i < factors; ++i) {
                Ap[threadIdx.x] += p[i] * YtY[i * factors + threadIdx.x];
            }

            for (int index = indptr[u]; index < indptr[u + 1]; ++index) {
                int i = indices[index];
                float confidence = data[index];
                dot(temp, &Y[i * factors], p);
                Ap[threadIdx.x] += (confidence - 1) * temp[0] * Y[i * factors + threadIdx.x];
            }

            // alpha = rsold / p.dot(Ap);
            dot(temp, p, Ap);

            float alpha = rsold[0] / temp[0];

            // x += alpha * p
            x[threadIdx.x] += alpha * p[threadIdx.x];

            // r -= alpha * Ap
            r[threadIdx.x] -= alpha * Ap[threadIdx.x];

            dot(rsnew, r, r);
            if (rsnew[0] < 1e-10) break;

            // p = r + (rsnew/rsold) * p
            p[threadIdx.x] = r[threadIdx.x] + (rsnew[0]/rsold[0]) * p[threadIdx.x];

            if (threadIdx.x == 0) {
                rsold[0] = rsnew[0];
            }

            // don't need to __syncthreads() here: will get called by something else
            // before it matters.
        }
    }
}

__global__ void l2_regularize_kernel(int factors, float regularization, float * YtY) {
    YtY[threadIdx.x * factors + threadIdx.x] += regularization;
}

#define CHECK_CUDA(code) { checkCuda((code), __FILE__, __LINE__); }
inline void checkCuda(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::stringstream err;
        err << "Cuda Error: " << cudaGetErrorString(code) << " (" << file << ":" << line << ")";
        throw std::runtime_error(err.str());
    }
}

const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "Unknown";
}

#define CHECK_CUBLAS(code) { checkCublas((code), __FILE__, __LINE__); }
inline void checkCublas(cublasStatus_t code, const char * file, int line) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        std::stringstream err;
        err << "cublas error: " << cublasGetErrorString(code)
            << " (" << file << ":" << line << ")";
        throw std::runtime_error(err.str());
    }
}

CudaDenseMatrix::CudaDenseMatrix(int rows, int cols, const float * host_data)
    : rows(rows), cols(cols) {
    CHECK_CUDA(cudaMalloc(&data, rows * cols * sizeof(float)));
    if (host_data) {
        CHECK_CUDA(cudaMemcpy(data, host_data, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void CudaDenseMatrix::to_host(float * out) const {
    CHECK_CUDA(cudaMemcpy(out, data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
}

CudaDenseMatrix::~CudaDenseMatrix() {
    CHECK_CUDA(cudaFree(data));
}

CudaCSRMatrix::CudaCSRMatrix(int rows, int cols, int nonzeros,
                             const int * indptr_, const int * indices_, const float * data_)
    : rows(rows), cols(cols), nonzeros(nonzeros) {

    CHECK_CUDA(cudaMalloc(&indptr, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(indptr, indptr_, (rows + 1)*sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&indices, nonzeros * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(indices, indices_, nonzeros * sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&data, nonzeros * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(data, data_, nonzeros * sizeof(int), cudaMemcpyHostToDevice));
}

CudaCSRMatrix::~CudaCSRMatrix() {
    CHECK_CUDA(cudaFree(indices));
    CHECK_CUDA(cudaFree(indptr));
    CHECK_CUDA(cudaFree(data));
}

CudaLeastSquaresSolver::CudaLeastSquaresSolver(int factors)
    : YtY(factors, factors, NULL) {
    CHECK_CUBLAS(cublasCreate(&blas_handle));
}

void CudaLeastSquaresSolver::least_squares(const CudaCSRMatrix & Cui,
                                           CudaDenseMatrix * X,
                                           const CudaDenseMatrix & Y,
                                           float regularization) const {
    int item_count = Y.rows, user_count = X->rows, factors = X->cols;
    if (X->cols != Y.cols) throw invalid_argument("X and Y should have the same number of columns");
    if (X->cols != YtY.cols) throw invalid_argument("Columns of X don't match number of factors");
    if (Cui.rows != X->rows) throw invalid_argument("Dimensionality mismatch between Cui and X");
    if (Cui.cols != Y.rows) throw invalid_argument("Dimensionality mismatch between Cui and Y");

    // calculate YtY: note this expects col-major (and we have row-major basically)
    // so that we're inverting the CUBLAS_OP_T/CU_BLAS_OP_N ordering to overcome
    // this (like calculate YYt instead of YtY)
    float alpha = 1.0, beta = 0.;
    CHECK_CUBLAS(cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             factors, factors, item_count,
                             &alpha,
                             Y.data, factors,
                             Y.data, factors,
                             &beta,
                             YtY.data, factors));
    CHECK_CUDA(cudaDeviceSynchronize());

    // regularize the matrix
    l2_regularize_kernel<<<1, factors>>>(factors, regularization, YtY.data);
    CHECK_CUDA(cudaDeviceSynchronize());

    // TODO: multi-gpu support
    int devId;
    CHECK_CUDA(cudaGetDevice(&devId));

    int multiprocessor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count,
                                      cudaDevAttrMultiProcessorCount,
                                      devId));

    int grid_size = 32 * multiprocessor_count;
    int thread_count = factors;
    int shared_memory_size = sizeof(float) * (3 * factors + 3);
    least_squares_cg_kernel<<<grid_size, thread_count, shared_memory_size>>>(
        factors, user_count, item_count,
        X->data, Y.data, YtY.data, Cui.indptr, Cui.indices, Cui.data);

    CHECK_CUDA(cudaDeviceSynchronize());
}

CudaLeastSquaresSolver::~CudaLeastSquaresSolver() {
    CHECK_CUBLAS(cublasDestroy(blas_handle));
}
}  // namespace implicit
