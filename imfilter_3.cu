#include <stdio.h>
#include <string.h>
#include <sys/times.h>
#include <sys/resource.h>
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

#define SIZE 32
#define KERNEL_SIZE 3
#define KS_DIV_2 (KERNEL_SIZE >> 1)
#define TILE_SIZE 12



__global__ void Kernel01(int size_filter, double *FilterMatrix, unsigned char* matrix_orig, int height, int width, unsigned char* matrix_filt) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row;
    int j = col;
    if (row < height && col < width) {
        float accumulator_red = 0;
        float accumulator_green = 0;
        float accumulator_blue = 0;
        for (size_t k = 0; k < size_filter; k++) { // kernel rows
            for (size_t l = 0; l < size_filter; l++) { // kernel elements/cols
                accumulator_red += FilterMatrix[k*size_filter + l] * (unsigned int) (matrix_orig[((i+k-1)*width + (j+l-1))*size_filter + 0]);
                accumulator_green += FilterMatrix[k*size_filter + l] * (unsigned int) (matrix_orig[((i+k-1)*width + (j+l-1))*size_filter + 1]);
                accumulator_blue += FilterMatrix[k*size_filter + l] * (unsigned int) (matrix_orig[((i+k-1)*width + (j+l-1))*size_filter + 2]);
            }
        }
        matrix_filt[(i*width + j)*size_filter + 0]= (unsigned int) accumulator_red;
        matrix_filt[(i*width + j)*size_filter + 1] = (unsigned int) accumulator_green;
        matrix_filt[(i*width + j)*size_filter + 2] = (unsigned int) accumulator_blue;
    }
}

__global__ void Kernel02(int size_filter, double *FilterMatrix, unsigned char* matrix_orig, int height, int width, unsigned char* matrix_filt)//(Matrix N, Matrix P)
{
    __shared__ unsigned char tileNs[SIZE][SIZE][3];
    // get thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // get the output indices
    int row_o = ty + blockIdx.y * TILE_SIZE;
    int col_o = tx + blockIdx.x * TILE_SIZE;

    // shift to obtain input indices
    int row_i = row_o - KS_DIV_2;
    int col_i = col_o - KS_DIV_2;

    // Load tile elements
    if(row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
        tileNs[ty][tx][0] = matrix_orig[(row_i*width + col_i)*size_filter + 0];
        tileNs[ty][tx][1] = matrix_orig[(row_i*width + col_i)*size_filter + 1];
        tileNs[ty][tx][2] = matrix_orig[(row_i*width + col_i)*size_filter + 2];
    }
    else {
        tileNs[ty][tx][0] = 0;
        tileNs[ty][tx][1] = 0;
        tileNs[ty][tx][2] = 0;
    }

    // Wait until all tile elements are loaded
    __syncthreads();

    // only compute if you're an output tile element
    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float accumulator_red = 0;
        float accumulator_green = 0;
        float accumulator_blue = 0;
        for(int y=0; y<size_filter; y++) {
            for(int x=0; x<size_filter; x++) {
                accumulator_red += FilterMatrix[y*size_filter + x] * tileNs[y+ty][x+tx][0];
                accumulator_green += FilterMatrix[y*size_filter + x] * tileNs[y+ty][x+tx][1];
                accumulator_blue += FilterMatrix[y*size_filter + x] * tileNs[y+ty][x+tx][2];
             }
         }
        // only write values if you are inside matrix bounds
        if(row_o < height && col_o < width) {
            matrix_filt[(row_o*width + col_o)*size_filter + 0]= (unsigned int) accumulator_red;
            matrix_filt[(row_o*width + col_o)*size_filter + 1] = (unsigned int) accumulator_green;
            matrix_filt[(row_o*width + col_o)*size_filter + 2] = (unsigned int) accumulator_blue;
        }
    }
}

void seq(int size_filter, double *FilterMatrix, unsigned char* matrix_orig, int height, int width, unsigned char* matrix_filt) {
    for (size_t i = size_filter/2; i < height-size_filter/2; i++) { // image row
        for (size_t j = size_filter/2; j < width-size_filter/2; j++) { // pixels in image row
            float accumulator_red = 0;
            float accumulator_green = 0;
            float accumulator_blue = 0;
            for (size_t k = 0; k < size_filter; k++) { // kernel rows
                for (size_t l = 0; l < size_filter; l++) { // kernel elements/cols
                    accumulator_red += FilterMatrix[k*size_filter + l] * (unsigned int) (matrix_orig[((i+k-1)*width + (j+l-1))*size_filter + 0]);
                    accumulator_green += FilterMatrix[k*size_filter + l] * (unsigned int) (matrix_orig[((i+k-1)*width + (j+l-1))*size_filter + 1]);
                    accumulator_blue += FilterMatrix[k*size_filter + l] * (unsigned int) (matrix_orig[((i+k-1)*width + (j+l-1))*size_filter + 2]);
                }
            }
            matrix_filt[(i*width + j)*size_filter + 0]= (unsigned int) accumulator_red;
            matrix_filt[(i*width + j)*size_filter + 1] = (unsigned int) accumulator_green;
            matrix_filt[(i*width + j)*size_filter + 2] = (unsigned int) accumulator_blue;

        }
    }
}

float GetTime(void);

int main (int argc, char *argv[])
{
    if (argc!=2) {
        printf("Usage: ./cuda image_to_be_filter\n");
        exit(1);
    }

    int width, height;
    int channels = 3;
    unsigned char *h_matrix_orig = stbi_load(argv[1], &width, &height, NULL, STBI_rgb);
    printf("La imagen es %d X %d\n", width, height);

    unsigned char *matrix_filt;
    matrix_filt = (unsigned char *) malloc (sizeof(unsigned char)*height*width*channels);
    double h_K[9] = {0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11};

    // Sequential
    float t1,t2;
    t1=GetTime();
    seq(channels, h_K, h_matrix_orig, width, height, matrix_filt);
    t2=GetTime();


    // Cuda 1
    float TiempoTotalCuda1, TiempoKernelCuda1;
    cudaEvent_t E0, E1, E2, E3;
    unsigned char *d_matrix_orig, *d_matrix_filt, *h_matrix_filt;
    h_matrix_filt = (unsigned char *) malloc (sizeof(unsigned char)*height*width*channels);
    double *d_K;

    int numBytesK = sizeof(h_K);
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
    //int SIZE = 32;
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

    Kernel01<<<dimGrid, dimBlock>>>(channels, d_K, d_matrix_orig, width, height, d_matrix_filt);

    cudaEventRecord(E2, 0);
    cudaEventSynchronize(E2);

    if (cudaSuccess != cudaGetLastError())
        printf("3: CUDA error at kernel exec: %s\n", cudaGetErrorString(cudaGetLastError()));
    err = cudaMemcpy(h_matrix_filt, d_matrix_filt, numBytesFilt, cudaMemcpyDeviceToHost);
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
        printf("5: CUDA error copying to Device: %s\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(d_matrix_orig2, h_matrix_orig, numBytesOrig, cudaMemcpyHostToDevice);
    if (err!=cudaSuccess) {
        printf("6: CUDA error copying to Device: %s\n", cudaGetErrorString(err));
    }


    cudaEventRecord(E5, 0);
    cudaEventSynchronize(E5);

    Kernel02<<<dimGrid, dimBlock>>>(channels, d_K2, d_matrix_orig2, width, height, d_matrix_filt2);



    cudaEventRecord(E6, 0);
    cudaEventSynchronize(E6);

    if (cudaSuccess != cudaGetLastError())
        printf("7: CUDA error at kernel exec: %s\n", cudaGetErrorString(cudaGetLastError()));
    //err = cudaThreadsSynchronize();
    //if (err!=cudaSuccess) {
    //    printf("8: CUDA error synchronizing threads: %s\n", cudaGetErrorString(err));
    //}
    err = cudaMemcpy(h_matrix_filt2, d_matrix_filt2, numBytesFilt, cudaMemcpyDeviceToHost);
    if (err!=cudaSuccess) {
        printf("9: CUDA error copying to Host: %s\n", cudaGetErrorString(err));
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

    //Size of the problem?
    //printf("Rendimiento Paralelo Global con Kernel01: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoTotalCuda1));
    //printf("Rendimiento Paralelo Kernel01: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoKernelCuda1));
    //printf("Rendimiento Secuencial: %4.2f GFLOPS\n", ((float) 5*N) / (1000000.0 * (t2 - t1)));


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

