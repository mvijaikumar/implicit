#ifndef IMPLICIT_CUDA_ALS_H_
#define IMPLICIT_CUDA_ALS_H_

// Forward ref: don't require the whole cublas definition here
struct cublasContext;

namespace implicit {
struct CudaCSRMatrix {
    CudaCSRMatrix(int rows, int cols, int nonzeros,
                  const int * indptr, const int * indices, const float * data);
    ~CudaCSRMatrix();
    int * indptr, * indices;
    float * data;
    int rows, cols, nonzeros;
};

struct CudaDenseMatrix {
    CudaDenseMatrix(int rows, int cols, const float * data);
    ~CudaDenseMatrix();

    void to_host(float * output) const;

    int rows, cols;
    float * data;
};

struct CudaLeastSquaresSolver {
    explicit CudaLeastSquaresSolver(int factors);
    ~CudaLeastSquaresSolver();

    void least_squares(const CudaCSRMatrix & Cui,
                       CudaDenseMatrix * X, const CudaDenseMatrix & Y,
                       float regularization) const;

    CudaDenseMatrix YtY;
    cublasContext * blas_handle;
};
}  // namespace implicit
#endif  // IMPLICIT_CUDA_ALS_H_
