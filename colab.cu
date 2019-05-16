%%cu
#include <stdio.h>
#include <string.h>
#include <sys/times.h>
#include <sys/resource.h>
#define STB_IMAGE_IMPLEMENTATION
#include "/content/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "/content/stb_image_write.h"

#define KERNEL_SIZE 5
#define PAD_KERNEL (KERNEL_SIZE /2)
#define TILE_SIZE 20
#define BLOCK_SIZE (TILE_SIZE + KERNEL_SIZE - 1)

__global__ void Kernel01(int size_filter, double *FilterMatrix, unsigned char* matrix_orig, int height, int width, unsigned char* matrix_filt, int channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row;
    int j = col;
    if ((row >= int(size_filter/2)) && (col >= int(size_filter/2)) && (row < (height-int(size_filter/2))) && (col <int((width-size_filter/2)))) {
        float accumulator_red = 0;
        float accumulator_green = 0;
        float accumulator_blue = 0;
        for (size_t k = 0; k < size_filter; k++) { // kernel rows
            for (size_t l = 0; l < size_filter; l++) { // kernel elements/cols
                accumulator_red += FilterMatrix[k*size_filter + l] * (unsigned int) (matrix_orig[((i+k-1)*width + (j+l-1))*channels + 0]);
                accumulator_green += FilterMatrix[k*size_filter + l] * (unsigned int) (matrix_orig[((i+k-1)*width + (j+l-1))*channels + 1]);
                accumulator_blue += FilterMatrix[k*size_filter + l] * (unsigned int) (matrix_orig[((i+k-1)*width + (j+l-1))*channels + 2]);
            }
        }
        matrix_filt[(i*width + j)*channels + 0]= (unsigned int) accumulator_red;
        matrix_filt[(i*width + j)*channels + 1] = (unsigned int) accumulator_green;
        matrix_filt[(i*width + j)*channels + 2] = (unsigned int) accumulator_blue;
    }
}

__global__ void Kernel02(int size_filter, double *FilterMatrix, unsigned char* matrix_orig, int height, int width, unsigned char* matrix_filt, int channels)//(Matrix N, Matrix P)
{
    __shared__ unsigned char tileNs[BLOCK_SIZE][BLOCK_SIZE][3];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = ty + blockIdx.y * TILE_SIZE;
    int col = tx + blockIdx.x * TILE_SIZE;
    int row_pad = row - PAD_KERNEL;
    int col_pad = col - PAD_KERNEL;
    if(row_pad >= 0 && row_pad < height && col_pad >= 0 && col_pad < width) {
        tileNs[ty][tx][0] = matrix_orig[(row_pad*width + col_pad)*channels + 0];
        tileNs[ty][tx][1] = matrix_orig[(row_pad*width + col_pad)*channels + 1];
        tileNs[ty][tx][2] = matrix_orig[(row_pad*width + col_pad)*channels + 2];
    }
    __syncthreads();

    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float accumulator_red = 0;
        float accumulator_green = 0;
        float accumulator_blue = 0;
        for(int k=0; k<size_filter; k++){
            for(int l=0; l<size_filter; l++){
                accumulator_red += FilterMatrix[k*size_filter + l] * tileNs[k+ty][l+tx][0];
                accumulator_green += FilterMatrix[k*size_filter + l] * tileNs[k+ty][l+tx][1];
                accumulator_blue += FilterMatrix[k*size_filter + l] * tileNs[k+ty][l+tx][2];
	    }
	}
        if(row < height && col < width) {
            matrix_filt[(row*width + col)*channels + 0]= (unsigned int) accumulator_red;
            matrix_filt[(row*width + col)*channels + 1] = (unsigned int) accumulator_green;
            matrix_filt[(row*width + col)*channels + 2] = (unsigned int) accumulator_blue;
        }
    }
}

void seq(int size_filter, double *FilterMatrix, unsigned char* matrix_orig, int height, int width, unsigned char* matrix_filt, int channels) {
    for (size_t i = size_filter/2; i < height-size_filter/2; i++) { // image row
        for (size_t j = size_filter/2; j < width-size_filter/2; j++) { // pixels in image row
            float accumulator_red = 0;
            float accumulator_green = 0;
            float accumulator_blue = 0;
            for (size_t k = 0; k < size_filter; k++) { // kernel rows
                for (size_t l = 0; l < size_filter; l++) { // kernel elements/cols
                    accumulator_red += FilterMatrix[k*size_filter + l] * (unsigned int) (matrix_orig[((i+k-1)*width + (j+l-1))*channels + 0]);
                    accumulator_green += FilterMatrix[k*size_filter + l] * (unsigned int) (matrix_orig[((i+k-1)*width + (j+l-1))*channels + 1]);
                    accumulator_blue += FilterMatrix[k*size_filter + l] * (unsigned int) (matrix_orig[((i+k-1)*width + (j+l-1))*channels + 2]);
                }
            }
            matrix_filt[(i*width + j)*channels + 0]= (unsigned int) accumulator_red;
            matrix_filt[(i*width + j)*channels + 1] = (unsigned int) accumulator_green;
            matrix_filt[(i*width + j)*channels + 2] = (unsigned int) accumulator_blue;

        }
    }
}

float GetTime(void);

int main (int argc, char *argv[])
{
    
    int width, height;
    int channels = 3;
    unsigned char *h_matrix_orig = stbi_load("image_barcelona_3072.jpg", &width, &height, NULL, STBI_rgb);
    printf("La imagen es %d X %d\n", width, height);
 
    unsigned char *matrix_filt;
    matrix_filt = (unsigned char *) malloc (sizeof(unsigned char)*height*width*channels);
    double h_K[25] = {0.0030,0.0133,0.0219,0.0133,0.0030,0.0133,0.0596,0.0983,0.0596,0.0133,0.0219,0.0983,0.1621,0.0983,0.0219,0.0133,0.0596,0.0983,0.0596,0.0133,0.0030,0.0133,0.0219,0.0133,0.0030};
    int size_filter = 5;

    // Sequential
    float t1,t2;
    t1=GetTime();
    seq(size_filter, h_K, h_matrix_orig, width, height, matrix_filt, channels);
    t2=GetTime();

 
    // Cuda 1
    float TiempoTotalCuda1, TiempoKernelCuda1;
    cudaEvent_t E0, E1, E2, E3;
    unsigned char *d_matrix_orig, *d_matrix_filt, *h_matrix_filt;
    h_matrix_filt = (unsigned char *) malloc (sizeof(unsigned char)*height*width*channels);
    double *d_K;

    int numBytesK = sizeof(double)*size_filter*size_filter;//sizeof(h_K);
    int numBytesOrig = sizeof(char)*height*width*channels;
    int numBytesFilt = sizeof(char)*height*width*channels;

    cudaEventCreate(&E0);
    cudaEventCreate(&E1);
    cudaEventCreate(&E2);
    cudaEventCreate(&E3);

    cudaEventRecord(E0, 0);
    cudaEventSynchronize(E0);

    // Obtener Memoria en el device
    cudaMalloc((double**)&d_K, numBytesK);
    cudaMalloc((unsigned char ***)&d_matrix_orig, numBytesOrig);
    cudaMalloc((unsigned char ***)&d_matrix_filt, numBytesFilt);

    // Copiar datos desde el host en el device
    cudaError err = cudaMemcpy(d_K, h_K, numBytesK, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("1: CUDA error copying to Device: %s\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(d_matrix_orig, h_matrix_orig, numBytesOrig, cudaMemcpyHostToDevice);
    if (err!=cudaSuccess) {
        printf("2: CUDA error copying to Device: %s\n", cudaGetErrorString(err));
    }

    // Ejecutar el kernel
    int SIZE = 32;
    int nThreads = SIZE;
    int N = width;
    int M = height;

    // numero de Blocks en cada dimension
    int nBlocksN = (N+nThreads-1)/nThreads;
    int nBlocksM = (M+nThreads-1)/nThreads;
    dim3 dimGrid(nBlocksM, nBlocksN, 1);
    dim3 dimBlock(nThreads, nThreads, 1);

    cudaEventRecord(E1, 0);
    cudaEventSynchronize(E1);
 

    Kernel01<<<dimGrid, dimBlock>>>(size_filter, d_K, d_matrix_orig, width, height, d_matrix_filt, channels);

    cudaEventRecord(E2, 0);
    cudaEventSynchronize(E2);

    if (cudaSuccess != cudaGetLastError())
        printf("3: CUDA error at kernel exec: %s\n", cudaGetErrorString(cudaGetLastError()));
    err = cudaMemcpy(h_matrix_filt, d_matrix_filt, numBytesFilt, cudaMemcpyDeviceToHost);
    printf(" %d %d %d %d %d %d", numBytesFilt, sizeof(h_matrix_filt), sizeof(d_matrix_filt), channels, height, width);
    if (err!=cudaSuccess) {
        printf("4: CUDA error copying to Host: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_K);
    cudaFree(d_matrix_orig);
    cudaFree(d_matrix_filt);

    cudaEventRecord(E3, 0);
    cudaEventSynchronize(E3);

    cudaEventElapsedTime(&TiempoTotalCuda1,  E0, E3);
    cudaEventElapsedTime(&TiempoKernelCuda1, E1, E2);
    cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

    // Cuda 2
    float TiempoTotalCuda2, TiempoKernelCuda2;
    cudaEvent_t E4, E5, E6, E7;
    unsigned char *d_matrix_orig2, *d_matrix_filt2, *h_matrix_filt2;
    h_matrix_filt2 = (unsigned char *) malloc (sizeof(unsigned char)*height*width*channels);
    double *d_K2;


    cudaEventCreate(&E4);
    cudaEventCreate(&E5);
    cudaEventCreate(&E6);
    cudaEventCreate(&E7);

    cudaEventRecord(E4, 0);
    cudaEventSynchronize(E4);

    // Obtener Memoria en el device
    cudaMalloc((double**)&d_K2, numBytesK);
    cudaMalloc((unsigned char ***)&d_matrix_orig2, numBytesOrig);
    cudaMalloc((unsigned char ***)&d_matrix_filt2, numBytesFilt);

    // Copiar datos desde el host en el device
    err = cudaMemcpy(d_K2, h_K, numBytesK, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("1: CUDA error copying to Device: %s\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(d_matrix_orig2, h_matrix_orig, numBytesOrig, cudaMemcpyHostToDevice);
    if (err!=cudaSuccess) {
        printf("2: CUDA error copying to Device: %s\n", cudaGetErrorString(err));
    }

    nThreads = BLOCK_SIZE;
    N = width;
    M = height;

    // numero de Blocks en cada dimension
    int nBlocksN2 = (N+nThreads-1)/nThreads;
    int nBlocksM2 = (M+nThreads-1)/nThreads;
    //dim3 dimGrid2(nBlocksM, nBlocksN, 1);
    //dim3 dimBlock2(nThreads, nThreads, 1);
    
    dim3 dimBlock2, dimGrid2;
    dimBlock2.x = BLOCK_SIZE, dimBlock2.y = BLOCK_SIZE, dimBlock2.z = 1;
    dimGrid2.x = ceil((float)width/TILE_SIZE),
    dimGrid2.y = ceil((float)height/TILE_SIZE),
    dimGrid2.z = 1;
    
    cudaEventRecord(E5, 0);
    cudaEventSynchronize(E5);

    Kernel02<<<dimGrid2, dimBlock2>>>(size_filter, d_K2, d_matrix_orig2, width, height, d_matrix_filt2, channels);

    cudaEventRecord(E6, 0);
    cudaEventSynchronize(E6);

    if (cudaSuccess != cudaGetLastError())
        printf("3: CUDA error at kernel exec: %s\n", cudaGetErrorString(cudaGetLastError()));
    err = cudaMemcpy(h_matrix_filt2, d_matrix_filt2, numBytesFilt, cudaMemcpyDeviceToHost);
    if (err!=cudaSuccess) {
        printf("4: CUDA error copying to Host: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_K2);
    cudaFree(d_matrix_orig2);
    cudaFree(d_matrix_filt2);

    cudaEventRecord(E7, 0);
    cudaEventSynchronize(E7);

    cudaEventElapsedTime(&TiempoTotalCuda2,  E4, E7);
    cudaEventElapsedTime(&TiempoKernelCuda2, E5, E6);
    cudaEventDestroy(E4); cudaEventDestroy(E5); cudaEventDestroy(E6); cudaEventDestroy(E7);

    printf("Dimensiones: %dx%d\n", N, M);
    printf("nThreads: %dx%d (%d)\n", nThreads, nThreads, nThreads * nThreads);
    printf("nBlocks: %dx%d (%d)\n", nBlocksN, nBlocksM, nBlocksN*nBlocksM);
    printf("Tiempo Secuencial: %4.6f milseg\n", t2-t1);
    printf("Tiempo Paralelo Global con Kernel01: %4.6f milseg\n", TiempoTotalCuda1);
    printf("Tiempo Paralelo Kernel01: %4.6f milseg\n", TiempoKernelCuda1);
    printf("Tiempo Paralelo Global con Kernel02: %4.6f milseg\n", TiempoTotalCuda2);
    printf("Tiempo Paralelo Kernel02: %4.6f milseg\n", TiempoKernelCuda2);

    printf("Rendimiento Secuencial: %4.2f GFLOPS\n", (((float) N*M*size_filter*size_filter) / (1000000.0 * (t2 - t1))));
    printf("Rendimiento Paralelo Global con Kernel01: %4.2f GFLOPS\n", (((float) N*M*size_filter*size_filter) / (1000000.0 * TiempoTotalCuda1)));
    printf("Rendimiento Paralelo Kernel01: %4.2f GFLOPS\n", (((float) N*M*size_filter*size_filter) / (1000000.0 * TiempoKernelCuda1)));
    printf("Rendimiento Paralelo Global con Kernel02: %4.2f GFLOPS\n", (((float) N*M*size_filter*size_filter) / (1000000.0 * TiempoTotalCuda2)));
    printf("Rendimiento Paralelo Kernel02: %4.2f GFLOPS\n", (((float) N*M*size_filter*size_filter) / (1000000.0 * TiempoKernelCuda2)));


    //Test
    stbi_write_jpg("SEQ.jpg", width, height, STBI_rgb, matrix_filt, 255);
    stbi_write_jpg("CUDA1.jpg", width, height, STBI_rgb, h_matrix_filt, 255);
    stbi_write_jpg("CUDA2.jpg", width, height, STBI_rgb, h_matrix_filt2, 255);


    free(h_matrix_orig);
    free(h_matrix_filt);
    free(h_matrix_filt2);
    free(matrix_filt);


    return 0;

}

float GetTime(void)        {
    struct timeval tim;
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    tim=ru.ru_utime;
    return ((double)tim.tv_sec + (double)tim.tv_usec / 1000000.0)*1000.0;
}
