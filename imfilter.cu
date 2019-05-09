#include <stdio.h>
#include <string.h>
#include <sys/times.h>
#include <sys/resource.h>
#include "lib/libbmp.h"


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




void img_to_matrix(bmp_img img, int height, int width, unsigned char* matrix) {
    for (size_t i = 0; i < height; i++) { // image row
        for (size_t j = 0; j < width; j++) { // pixels in image row
            matrix[(i*width +j)*3 + 0] = img.img_pixels[i][j].red;
            matrix[(i*width +j)*3 + 1] = img.img_pixels[i][j].green;
            matrix[(i*width +j)*3 + 2] = img.img_pixels[i][j].blue;
        }


    }
}

void matrix_to_img(unsigned char* matrix, int height, int width, bmp_img img) {
    for (size_t i = 0; i < height; i++) { // image row
        for (size_t j = 0; j < width; j++) { // pixels in image row
            img.img_pixels[i][j].red = matrix[(i*width +j)*3 + 0];
            img.img_pixels[i][j].green = matrix[(i*width +j)*3 + 1];
            img.img_pixels[i][j].blue = matrix[(i*width +j)*3 + 2];
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

    bmp_img h_img_orig;
    bmp_img_read(&h_img_orig, argv[1]);
    int width = (int) h_img_orig.img_header.biWidth;
    int height = (int) h_img_orig.img_header.biHeight;
    int channels = 3;
    printf("La imagen es %d X %d\n", width, height);

    unsigned char *h_matrix_orig;
    h_matrix_orig = (unsigned char *) malloc (sizeof(unsigned char)*height*width*channels);

    img_to_matrix(h_img_orig, height, width, h_matrix_orig);


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

    printf("Dimensiones: %dx%d\n", N, M);
    printf("nThreads: %dx%d (%d)\n", nThreads, nThreads, nThreads * nThreads);
    printf("nBlocks: %dx%d (%d)\n", nBlocksN, nBlocksM, nBlocksN*nBlocksM);
    printf("Tiempo Secuencial: %4.6f milseg\n", t2-t1);
    printf("Tiempo Paralelo Global con Kernel01: %4.6f milseg\n", TiempoTotalCuda1);
    printf("Tiempo Paralelo Kernel01: %4.6f milseg\n", TiempoKernelCuda1);

    //Size of the problem?
    //printf("Rendimiento Paralelo Global con Kernel01: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoTotalCuda1));
    //printf("Rendimiento Paralelo Kernel01: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoKernelCuda1));
    //printf("Rendimiento Secuencial: %4.2f GFLOPS\n", ((float) 5*N) / (1000000.0 * (t2 - t1)));


    //Test
    bmp_img img_filt;
    bmp_img_init_df(&img_filt, height, width);
    matrix_to_img(matrix_filt, height, width, img_filt);
    bmp_img_write(&img_filt, strcat("SEQ", argv[1]));

    bmp_img h_img_filt;
    bmp_img_init_df(&h_img_filt, height, width);
    matrix_to_img(h_matrix_filt, height, width, h_img_filt);
    bmp_img_write(&h_img_filt, strcat("CUDA1", argv[1]));

    free(h_matrix_orig);
    free(h_matrix_filt);
    free(matrix_filt);


    bmp_img_free(&h_img_orig);
    bmp_img_free(&h_img_filt);
    bmp_img_free(&img_filt);




    return 0;

}

float GetTime(void)        {
    struct timeval tim;
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    tim=ru.ru_utime;
    return ((double)tim.tv_sec + (double)tim.tv_usec / 1000000.0)*1000.0;
}
