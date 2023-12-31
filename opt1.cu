#include <cusparse_v2.h>
#include <cublas_v2.h>

#define gpuCheck(stmt)                                               \
  do {                                                               \
      cudaError_t err = stmt;                                        \
      if (err != cudaSuccess) {                                      \
          printf("%s. Failed to run stmt %s\n",	cudaGetErrorName(err), #stmt);           \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuBLAS status
#define cublasCheck(stmt)                                            \
  do {                                                               \
      cublasStatus_t err = stmt;                                     \
      if (err != CUBLAS_STATUS_SUCCESS) {                            \
          printf("ERROR. Failed to run cuBLAS stmt %s\n", #stmt);    \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuSPARSE status
#define cusparseCheck(stmt)                                          \
  do {                                                               \
      cusparseStatus_t err = stmt;                                   \
      if (err != CUSPARSE_STATUS_SUCCESS) {                          \
          printf("ERROR. Failed to run cuSPARSE stmt %s\n", #stmt);  \
          break;                                                     \
      }                                                              \
  } while (0)

long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}

// -----------
// GLOBAL VARS
// -----------

const float zero = 0.0f;
const float one = 1.0f;

// Differentation matrices
float *dX, *dY, *dZ;
// Note: dX and dY will have same RowPtr and ColIndx
int *dXYRowPtr, *dZRowPtr;
int *dXYColIndx, *dZColIndx;
// Device data (use ping pong strategy for input/output temp)
float *t_d[2], *p_d;
// cuSPARSE handle
cusparseHandle_t cusparseHandle;
// cuSPARSE diff matrix descriptions
cusparseSpMatDescr_t dXDescr, dYDescr, dZDescr;
// cuSPARSE data matrix descriptions (note ping pong strategy is used for T descr)
cusparseDnMatDescr_t tXDescr[2], tYDescr[2], tZDescr[2];
// cuSPARSE calculation buffers
size_t bufferSizeX, bufferSizeY, bufferSizeZ;
float *bufferX, *bufferY, *bufferZ;
// cuBLAS handle
cublasHandle_t cublasHandle;

// ------------
// CALCULATIONS
// ------------

// Differentiation matrix, stencil is adapted for orientation
void diffMatrixInit(float* A, int* ArowPtr, int* AcolIndx,
    int rows, float stencil[3]) {
  // Variable holding the position to insert a new element
  size_t ptr = 0;
  ArowPtr[0] = ptr;

  // Configure first row (2 elements due to boundary)
  A[ptr] = stencil[0] + stencil[1];
  AcolIndx[ptr++] = 0;
  A[ptr] = stencil[2];
  AcolIndx[ptr++] = 1;
  ArowPtr[1] = ptr;

  // Fill middle of the matrix
  for (int i = 1; i < (rows - 1); ++i) {
    for (int k = 0; k < 3; ++k) {
      A[ptr] = stencil[k];
      AcolIndx[ptr++] = i + k - 1;
    }
    ArowPtr[i + 1] = ptr;
  }

  // Configure last row (2 elements due to boundary)
  A[ptr] = stencil[0];
  AcolIndx[ptr++] = rows - 2;
  A[ptr] = stencil[1] + stencil[2];
  AcolIndx[ptr++] = rows - 1;
  ArowPtr[rows] = ptr;
}

void cusparseDiffMatConfig(int nx, int ny, int nz, int nzv_xy, int nzv_z) {
    cusparseCheck(cusparseCreateCsr(&dXDescr, nx, nx, nzv_xy,
            dXYRowPtr, dXYColIndx, dX,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F));
    cusparseCheck(cusparseCreateCsr(&dYDescr, ny, ny, nzv_xy,
            dXYRowPtr, dXYColIndx, dY,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F));
    cusparseCheck(cusparseCreateCsr(&dZDescr, nz, nz, nzv_z,
            dZRowPtr, dZColIndx, dZ,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F));

    // dummy strided batches
    cusparseCheck(cusparseCsrSetStridedBatch(dXDescr, nz, 0, 0));
    cusparseCheck(cusparseCsrSetStridedBatch(dYDescr, nz, 0, 0));
}

void cusparseDataMatConfig(int nx, int ny, int nz) {
    // X matrix has col major layout due to later transposition in calc
    cusparseCheck(cusparseCreateDnMat(&tXDescr[0], nx, ny, ny,
            t_d[0], CUDA_R_32F, CUSPARSE_ORDER_COL));
    cusparseCheck(cusparseCreateDnMat(&tXDescr[1], nx, ny, ny,
            t_d[1], CUDA_R_32F, CUSPARSE_ORDER_COL));
    // Y matrix is normal
    cusparseCheck(cusparseCreateDnMat(&tYDescr[0], ny, nx, nx,
            t_d[0], CUDA_R_32F, CUSPARSE_ORDER_ROW));
    cusparseCheck(cusparseCreateDnMat(&tYDescr[1], ny, nx, nx,
            t_d[1], CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // Flattened version of data for operation by the diff Z matrix
    cusparseCheck(cusparseCreateDnMat(&tZDescr[0], nz, nx * ny, nx * ny,
            t_d[0], CUDA_R_32F, CUSPARSE_ORDER_ROW)); 
    cusparseCheck(cusparseCreateDnMat(&tZDescr[1], nz, nx * ny, nx * ny,
            t_d[1], CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // Use strided batches to create XY matrix for all layers
    cusparseCheck(cusparseDnMatSetStridedBatch(tXDescr[0], nz, nx * ny));
    cusparseCheck(cusparseDnMatSetStridedBatch(tXDescr[1], nz, nx * ny));
    cusparseCheck(cusparseDnMatSetStridedBatch(tYDescr[0], nz, nx * ny));
    cusparseCheck(cusparseDnMatSetStridedBatch(tYDescr[1], nz, nx * ny));
}

// Calculate and allocate calculation buffer
void cusparseCalcBufferAlloc() {
    // Note: calculation buffers will be reused as ping pong buffers
    // Note: Buffer size symmetrical between t___[0] and t___[1],
    //       hence only one calc performed 
    cusparseCheck(cusparseSpMM_bufferSize(cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, dXDescr, tXDescr[0],
                    &zero, tXDescr[1],
                    CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1,
                    &bufferSizeX)
                );
    cusparseCheck(cusparseSpMM_bufferSize(cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, dYDescr, tYDescr[0],
                    &one, tYDescr[1],
                    CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2,
                    &bufferSizeY)
                );
    cusparseCheck(cusparseSpMM_bufferSize(cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, dZDescr, tZDescr[0],
                    &one, tZDescr[1],
                    CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2,
                    &bufferSizeZ)
                );
    
    gpuCheck(cudaMallocManaged(&bufferX, bufferSizeX));
    gpuCheck(cudaMallocManaged(&bufferY, bufferSizeY));
    gpuCheck(cudaMallocManaged(&bufferZ, bufferSizeZ));
}

//preprocessing
void cusparseCalcPreprocess() {
    cusparseCheck(cusparseSpMM_preprocess(cusparseHandle, 
                    CUSPARSE_OPERATION_NON_TRANSPOSE, 
                    CUSPARSE_OPERATION_NON_TRANSPOSE, 
                    &one, dXDescr, tXDescr[0], 
                    &one, tXDescr[1], 
                    CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, 
                    bufferX));
    cusparseCheck(cusparseSpMM_preprocess(cusparseHandle, 
                    CUSPARSE_OPERATION_NON_TRANSPOSE, 
                    CUSPARSE_OPERATION_NON_TRANSPOSE, 
                    &one, dYDescr, tYDescr[0], 
                    &one, tYDescr[1], 
                    CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, 
                    bufferY));
    cusparseCheck(cusparseSpMM_preprocess(cusparseHandle, 
                    CUSPARSE_OPERATION_NON_TRANSPOSE, 
                    CUSPARSE_OPERATION_NON_TRANSPOSE, 
                    &one, dZDescr, tZDescr[0], 
                    &one, tZDescr[1], 
                    CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, 
                    bufferZ));
}


// Perform cuSPARSE part of calculation
void cusparseCalc(int in) {
    int out = !in;

    cusparseCheck(cusparseSpMM(cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, dXDescr, tXDescr[in],
                    &zero, tXDescr[out],
                    CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1,
                    bufferX)
                );
    cudaDeviceSynchronize();

    cusparseCheck(cusparseSpMM(cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, dYDescr, tYDescr[in],
                    &one, tYDescr[out],
                    CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2,
                    bufferY)
                );
    cudaDeviceSynchronize();

    cusparseCheck(cusparseSpMM(cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, dZDescr, tZDescr[in],
                    &one, tZDescr[out],
                    CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2,
                    bufferZ)
                );
    cudaDeviceSynchronize();
}

// Perform cuBLAS part of calculation
void cublasCalc(int out, int nx, int ny, int nz, float sdc) {
    cublasCheck(cublasSaxpy(cublasHandle, nx * ny * nz,
                    &sdc, p_d, 1, t_d[out], 1));
    cudaDeviceSynchronize();       
}

__global__ void addAmbTemp(float* tOut, float ct, float ambTemp) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    tOut[id] += ct * ambTemp;
}

void hotspot_opt1(float *p, float *tIn, float *tOut,
        int nx, int ny, int nz,
        float Cap, 
        float Rx, float Ry, float Rz, 
        float dt, int numiter) 
{
    long long start_setup = get_time();

    float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

    // Copy temp and power data to device
    size_t s = sizeof(float) * nx * ny * nz;  
    gpuCheck(cudaMalloc(&p_d,s));
    gpuCheck(cudaMalloc(&t_d[0],s));
    gpuCheck(cudaMalloc(&t_d[1],s));
    gpuCheck(cudaMemcpy(t_d[0], tIn, s, cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(p_d, p, s, cudaMemcpyHostToDevice));

    // non-zero values for respective matrices
    int nzv_xy = 3 * (nx - 2) + 4;
    int nzv_z = 3 * (nz - 2) + 4;

    gpuCheck(cudaMallocManaged(&dX, sizeof(float) * nzv_xy));
    gpuCheck(cudaMallocManaged(&dY, sizeof(float) * nzv_xy));
    gpuCheck(cudaMallocManaged(&dZ, sizeof(float) * nzv_z));
    gpuCheck(cudaMallocManaged(&dXYRowPtr, sizeof(int) * (nx + 1)));
    gpuCheck(cudaMallocManaged(&dZRowPtr, sizeof(int) * (nz + 1)));
    gpuCheck(cudaMallocManaged(&dXYColIndx, sizeof(int) * nzv_xy));
    gpuCheck(cudaMallocManaged(&dZColIndx, sizeof(int) * nzv_z));
    
    // Create CSR diff matrices
    // TODO: optimise by only adding cc component once
    float stencilX[3] = {cw, cc, ce};
    float stencilY[3] = {cn, 0.0f, cs};
    float stencilZ[3] = {cb, 0.0f, ct};
    diffMatrixInit(dX, dXYRowPtr, dXYColIndx, nx, stencilX);
    diffMatrixInit(dY, dXYRowPtr, dXYColIndx, ny, stencilY);
    diffMatrixInit(dZ, dZRowPtr, dZColIndx, nz, stencilZ);

    // Init cuSPARSE
    cusparseCheck(cusparseCreate(&cusparseHandle));

    // Set up cuSPARSE matrices and calc buffers
    cusparseDiffMatConfig(nx, ny, nz, nzv_xy, nzv_z);
    cusparseDataMatConfig(nx, ny, nz);
    cusparseCalcBufferAlloc();
    cusparseCalcPreprocess();

    // Init cuBLAS
    cublasCheck(cublasCreate(&cublasHandle));
    cublasCheck(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));

    long long stop_setup = get_time();
    float time_setup = (float)((stop_setup - start_setup)/(1000.0 * 1000.0));
    printf("Time for setup: %.3f (s)\n",time_setup);

    long long start = get_time();
    int in = 0;
    dim3 blockDim(1024);
    dim3 gridDim((nx * ny * nz + 1024) / 1024);
    float amb_temp = 80.0f;
    for (int i = 0; i < numiter; ++i) {
        cusparseCalc(in);
        cublasCalc(!in, nx, ny, nz, stepDivCap);
        addAmbTemp<<<gridDim, blockDim>>>(t_d[!in], ct, amb_temp);
        cudaDeviceSynchronize();
        // Swap ping-pong buffers
        in = !in;
    }
    int out = in;
    
    long long stop = get_time();
    float time = (float)((stop - start)/(1000.0 * 1000.0));
    printf("Time: %.3f (s)\n",time);    
    cudaMemcpy(tOut, t_d[out], s, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(p_d);
    cudaFree(t_d[0]);
    cudaFree(t_d[1]);
    return;
}

