%%cu
#include <stdio.h>
#include <stdlib.h>

#ifndef __LIBBMP_H__
#define __LIBBMP_H__

#define BMP_MAGIC 19778

#define BMP_GET_PADDING(a) ((a) % 4)


#include <stdio.h>
#include <stdlib.h>

enum bmp_error
{
	BMP_FILE_NOT_OPENED = -4,
	BMP_HEADER_NOT_INITIALIZED,
	BMP_INVALID_FILE,
	BMP_ERROR,
	BMP_OK = 0
};

typedef struct _bmp_header
{
	unsigned int   bfSize;
	unsigned int   bfReserved;
	unsigned int   bfOffBits;
	
	unsigned int   biSize;
	int            biWidth;
	int            biHeight;
	unsigned short biPlanes;
	unsigned short biBitCount;
	unsigned int   biCompression;
	unsigned int   biSizeImage;
	int            biXPelsPerMeter;
	int            biYPelsPerMeter;
	unsigned int   biClrUsed;
	unsigned int   biClrImportant;
} bmp_header;

typedef struct _bmp_pixel
{
	unsigned char blue;
	unsigned char green;
	unsigned char red;
} bmp_pixel;

// This is faster than a function call
#define BMP_PIXEL(r,g,b) ((bmp_pixel){(b),(g),(r)})

typedef struct _bmp_img
{
	bmp_header   img_header;
	bmp_pixel  **img_pixels;
} bmp_img;

// BMP_HEADER
void            bmp_header_init_df             (bmp_header*,
                                                const int,
                                                const int);

enum bmp_error  bmp_header_write               (const bmp_header*,
                                                FILE*);

enum bmp_error  bmp_header_read                (bmp_header*,
                                                FILE*);

// BMP_PIXEL
void            bmp_pixel_init                 (bmp_pixel*,
                                                const unsigned char,
                                                const unsigned char,
                                                const unsigned char);

// BMP_IMG
void            bmp_img_alloc                  (bmp_img*);
void            bmp_img_init_df                (bmp_img*,
                                                const int,
                                                const int);
void            bmp_img_free                   (bmp_img*);

enum bmp_error  bmp_img_write                  (const bmp_img*,
                                                const char*);

enum bmp_error  bmp_img_read                   (bmp_img*,
                                                const char*);

#endif /* __LIBBMP_H__ */


// BMP_HEADER

void bmp_header_init_df (bmp_header *header, const int width, const int height)
{
	header->bfSize = (sizeof (bmp_pixel) * width + BMP_GET_PADDING (width)) * abs (height);
	header->bfReserved = 0;
	header->bfOffBits = 54;
	header->biSize = 40;
	header->biWidth = width;
	header->biHeight = height;
	header->biPlanes = 1;
	header->biBitCount = 24;
	header->biCompression = 0;
	header->biSizeImage = 0;
	header->biXPelsPerMeter = 0;
	header->biYPelsPerMeter = 0;
	header->biClrUsed = 0;
	header->biClrImportant = 0;
}

enum bmp_error bmp_header_write (const bmp_header *header, FILE *img_file)
{
	if (header == NULL)
	{
		return BMP_HEADER_NOT_INITIALIZED; 
	}
	else if (img_file == NULL)
	{
		return BMP_FILE_NOT_OPENED;
	}
	
	// Since an adress must be passed to fwrite, create a variable!
	const unsigned short magic = BMP_MAGIC;
	fwrite (&magic, sizeof (magic), 1, img_file);
	
	// Use the type instead of the variable because its a pointer!
	fwrite (header, sizeof (bmp_header), 1, img_file);
	return BMP_OK;
}

enum bmp_error bmp_header_read (bmp_header *header, FILE *img_file)
{
	if (img_file == NULL)
	{
		return BMP_FILE_NOT_OPENED;
	}
	
	// Since an adress must be passed to fread, create a variable!
	unsigned short magic;
	
	// Check if its an bmp file by comparing the magic nbr:
	if (fread (&magic, sizeof (magic), 1, img_file) != 1 ||
	    magic != BMP_MAGIC)
	{
		return BMP_INVALID_FILE;
	}
	
	if (fread (header, sizeof (bmp_header), 1, img_file) != 1)
	{
		return BMP_ERROR;
	}

	return BMP_OK;
}

// BMP_PIXEL

void bmp_pixel_init (bmp_pixel *pxl,
                const unsigned char  red,
                const unsigned char  green,
                const unsigned char  blue)
{
	pxl->red = red;
	pxl->green = green;
	pxl->blue = blue;
}

// BMP_IMG

void bmp_img_alloc (bmp_img *img)
{
	const size_t h = abs (img->img_header.biHeight);
	
	// Allocate the required memory for the pixels:
	img->img_pixels = (bmp_pixel  **)malloc (sizeof (bmp_pixel*) * h);
	
	for (size_t y = 0; y < h; y++)
	{
		img->img_pixels[y] = (bmp_pixel  *)malloc (sizeof (bmp_pixel) * img->img_header.biWidth);
	}
}

void bmp_img_init_df (bmp_img   *img, const int  width, const int  height)
{
	// INIT the header with default values:
	bmp_header_init_df (&img->img_header, width, height);
	bmp_img_alloc (img);
}

void bmp_img_free (bmp_img *img)
{
	const size_t h = abs (img->img_header.biHeight);
	
	for (size_t y = 0; y < h; y++)
	{
		free (img->img_pixels[y]);
	}
	free (img->img_pixels);
}

enum bmp_error bmp_img_write (const bmp_img *img, const char *filename)
{
	FILE *img_file = fopen (filename, "wb");
	
	if (img_file == NULL)
	{
		return BMP_FILE_NOT_OPENED;
	}
	
	// NOTE: This way the correct error code could be returned.
	const enum bmp_error err = bmp_header_write (&img->img_header, img_file);
	
	if (err != BMP_OK)
	{
		// ERROR: Could'nt write the header!
		fclose (img_file);
		return err;
	}
	
	// Select the mode (bottom-up or top-down):
	const size_t h = abs (img->img_header.biHeight);
	const size_t offset = (img->img_header.biHeight > 0 ? h - 1 : 0);
	
	// Create the padding:
	const unsigned char padding[3] = {'\0', '\0', '\0'};
	
	// Write the content:
	for (size_t y = 0; y < h; y++)
	{
		// Write a whole row of pixels to the file:
		fwrite (img->img_pixels[ (offset - y)], sizeof (bmp_pixel), img->img_header.biWidth, img_file);
		
		// Write the padding for the row!
		fwrite (padding, sizeof (unsigned char), BMP_GET_PADDING (img->img_header.biWidth), img_file);
	}
	
	// NOTE: All good!
	fclose (img_file);
	return BMP_OK;
}

enum bmp_error bmp_img_read (bmp_img *img, const char *filename)
{
	FILE *img_file = fopen (filename, "rb");
	
	if (img_file == NULL)
	{
		return BMP_FILE_NOT_OPENED;
	}
	
	// NOTE: This way the correct error code can be returned.
	const enum bmp_error err = bmp_header_read (&img->img_header, img_file);
	
	if (err != BMP_OK)
	{
		// ERROR: Could'nt read the image header!
		fclose (img_file);
		return err;
	}
	
	bmp_img_alloc (img);
	
	// Select the mode (bottom-up or top-down):
	const size_t h = abs (img->img_header.biHeight);
	const size_t offset = (img->img_header.biHeight > 0 ? h - 1 : 0);
	const size_t padding = BMP_GET_PADDING (img->img_header.biWidth);
	
	// Needed to compare the return value of fread
	const size_t items = img->img_header.biWidth;
	
	// Read the content:
	for (size_t y = 0; y < h; y++)
	{
		// Read a whole row of pixels from the file:
		if (fread (img->img_pixels[ (offset - y)], sizeof (bmp_pixel), items, img_file) != items)
		{
			fclose (img_file);
			return BMP_ERROR;
		}
		
		// Skip the padding:
		fseek (img_file, padding, SEEK_CUR);
	}
	
	// NOTE: All good!
	fclose (img_file);
	return BMP_OK;
}






bmp_img FilterSeq (int size_filter, double FilterMatrix[3][3], bmp_img img_orig, int height, int width) {
    bmp_img img_filt;
    bmp_img_init_df(&img_filt, height, width);
    for (size_t i = 0; i < height; i++) { // image row
        for (size_t j = 0; j < width; j++) { // pixels in image row
            double accumulator_red = 0;
            double accumulator_green = 0;
            double accumulator_blue = 0;
            for (size_t k = 0; k < size_filter; k++) { // kernel rows
                for (size_t l = 0; l < size_filter; l++) { // kernel elements/cols
                    if ((i % size_filter == k) && (j % size_filter == l)) {// corresponding element
                        accumulator_red += FilterMatrix[k][l] * (double) (img_orig.img_pixels[i][j].red);
                        accumulator_green += FilterMatrix[k][l] * (double) (img_orig.img_pixels[i][j].green);
                        accumulator_blue += FilterMatrix[k][l] * (double) (img_orig.img_pixels[i][j].blue);
                    }
                }
            }
            img_filt.img_pixels[i][j].red = (int) accumulator_red;
            img_filt.img_pixels[i][j].green = (int) accumulator_green;
            img_filt.img_pixels[i][j].blue = (int) accumulator_blue;
        }
    }
    return img_filt;
}

__global__ void Kernel01(int size_filter, double FilterMatrix[3][3], bmp_img img_orig, int height, int width, bmp_img img_filt) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row;
    int j = col;
    if (row < height && col < width) {
    //bmp_img img_filt;
    //bmp_img_init_df(&img_filt, height, width);
    //for (size_t i = 0; i < height; i++) { // image row
    //    for (size_t j = 0; j < width; j++) { // pixels in image row
            double accumulator_red = 0;
            double accumulator_green = 0;
            double accumulator_blue = 0;
            for (size_t k = 0; k < size_filter; k++) { // kernel rows
                for (size_t l = 0; l < size_filter; l++) { // kernel elements/cols
                    if ((i % size_filter == k) && (j % size_filter == l)) {// corresponding element
                        accumulator_red += FilterMatrix[k][l] * (double) (img_orig.img_pixels[i][j].red);
                        accumulator_green += FilterMatrix[k][l] * (double) (img_orig.img_pixels[i][j].green);
                        accumulator_blue += FilterMatrix[k][l] * (double) (img_orig.img_pixels[i][j].blue);
                    }
                }
            }
            img_filt.img_pixels[i][j].red = (int) accumulator_red;
            img_filt.img_pixels[i][j].green = (int) accumulator_green;
            img_filt.img_pixels[i][j].blue = (int) accumulator_blue;
     //   }
    //}
    //return img_filt;
    }
}


void img_to_matrix(bmp_img img, int height, int width, unsigned char*** matrix) {
    for (size_t i = 0; i < height; i++) { // image row
        for (size_t j = 0; j < width; j++) { // pixels in image row
               matrix[i][j][0] = img.img_pixels[i][j].red;
               matrix[i][j][1] = img.img_pixels[i][j].green;
               matrix[i][j][2] = img.img_pixels[i][j].blue;
          }
                                               
                
    }
}

void matrix_to_img(unsigned char*** matrix, int height, int width, bmp_img img) {
    for (size_t i = 0; i < height; i++) { // image row
        for (size_t j = 0; j < width; j++) { // pixels in image row
               img.img_pixels[i][j].red = matrix[i][j][0];
               img.img_pixels[i][j].green = matrix[i][j][1];
               img.img_pixels[i][j].blue = matrix[i][j][2];
          }
    }
}


__global__ void Kernel012(int size_filter, double *FilterMatrix, unsigned char* matrix_orig, int height, int width, unsigned char* matrix_filt) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row;
    int j = col;
    if (row < height && col < width) {
    //bmp_img img_filt;
    //bmp_img_init_df(&img_filt, height, width);
    //for (size_t i = 0; i < height; i++) { // image row
    //    for (size_t j = 0; j < width; j++) { // pixels in image row
            double accumulator_red = 0;
            double accumulator_green = 0;
            double accumulator_blue = 0;
            for (size_t k = 0; k < size_filter; k++) { // kernel rows
                for (size_t l = 0; l < size_filter; l++) { // kernel elements/cols
                    if ((i % size_filter == k) && (j % size_filter == l)) {// corresponding element
                        accumulator_red += FilterMatrix[k*size_filter + l] * (double) (matrix_orig[(i*width + j)*size_filter + 0]);
                        accumulator_green += FilterMatrix[k*size_filter + l] * (double) (matrix_orig[(i*width + j)*size_filter + 1]);
                        accumulator_blue += FilterMatrix[k*size_filter + l] * (double) (matrix_orig[(i*width + j)*size_filter + 2]);
                    }
                }
            }
            matrix_filt[(i*width + j)*size_filter + 0]= (int) accumulator_red;
            matrix_filt[(i*width + j)*size_filter + 1] = (int) accumulator_green;
            matrix_filt[(i*width + j)*size_filter + 2] = (int) accumulator_blue;
     //   }
    //}
    //return img_filt;
    }
}




void img_to_matrix2(bmp_img img, int height, int width, unsigned char* matrix) {
    for (size_t i = 0; i < height; i++) { // image row
        for (size_t j = 0; j < width; j++) { // pixels in image row
               matrix[(i*width +j)*3 + 0] = img.img_pixels[i][j].red;
               matrix[(i*width +j)*3 + 1] = img.img_pixels[i][j].green;
               matrix[(i*width +j)*3 + 2] = img.img_pixels[i][j].blue;
          }
                                               
                
    }
}

void matrix_to_img2(unsigned char* matrix, int height, int width, bmp_img img) {
    for (size_t i = 0; i < height; i++) { // image row
        for (size_t j = 0; j < width; j++) { // pixels in image row
               img.img_pixels[i][j].red = matrix[(i*width +j)*3 + 0];
               img.img_pixels[i][j].green = matrix[(i*width +j)*3 + 1];
               img.img_pixels[i][j].blue = matrix[(i*width +j)*3 + 2];
          }
    }
}


void seq(int size_filter, double *FilterMatrix, unsigned char* matrix_orig, int height, int void seq(int size_filter, double *FilterMatrix, unsigned char* matrix_orig, int height, int width, unsigned char* matrix_filt) {
    //int row = blockIdx.y * blockDim.y + threadIdx.y;
    //int col = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = row;
    //int j = col;
    //if (row < height && col < width) {
    //bmp_img img_filt;
    //bmp_img_init_df(&img_filt, height, width);
    for (size_t i = size_filter/2; i < height-size_filter/2; i++) { // image row
        for (size_t j = size_filter/2; j < width-size_filter/2; j++) { // pixels in image row
            float accumulator_red = 0;
            float accumulator_green = 0;
            float accumulator_blue = 0;
            //int count = 0;
            // position mask:
            //def f(i,j,k,l):
            //    return (i+k-1,j+l-1)
            for (size_t k = 0; k < size_filter; k++) { // kernel rows
                for (size_t l = 0; l < size_filter; l++) { // kernel elements/cols
                    //if ((i % size_filter == k) && (j % size_filter == l)) {// corresponding element
                        accumulator_red += FilterMatrix[k*size_filter + l] * (unsigned int) (matrix_orig[((i+k-1)*width + (j+l-1))*size_filter + 0]);
                        accumulator_green += FilterMatrix[k*size_filter + l] * (unsigned int) (matrix_orig[((i+k-1)*width + (j+l-1))*size_filter + 1]);
                        accumulator_blue += FilterMatrix[k*size_filter + l] * (unsigned int) (matrix_orig[((i+k-1)*width + (j+l-1))*size_filter + 2]);
                        //printf("%d %f %f %d \n", (unsigned int) accumulator_red, accumulator_red, FilterMatrix[k*size_filter + l], (int) (matrix_orig[(i*width + j)*size_filter + 0]));
                        //count += 1;
                    //}
                }
            }
            //printf("%d\n", count);
            matrix_filt[(i*width + j)*size_filter + 0]= (unsigned int) accumulator_red;
            matrix_filt[(i*width + j)*size_filter + 1] = (unsigned int) accumulator_green;
            matrix_filt[(i*width + j)*size_filter + 2] = (unsigned int) accumulator_blue;
            
        }
    }
    //return img_filt;
    //}
}

int main (int argc, char *argv[])
{
    // seq
    /*
    bmp_img img_orig;
	  bmp_img_read(&img_orig, "sample_crop.bmp");
    double K[3][3] = {0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11}; // mean filter
    bmp_img img_filt  = FilterSeq (3,K, img_orig, 216, 216);
    bmp_img_write(&img_filt, "sample_filtered.bmp");
    bmp_img_free(&img_orig);
    bmp_img_free(&img_filt);
    bmp_img_free(&img_filt2);
    */
      
    
    // cuda 1
    //bmp_img h_img_orig, h_img_filt;
    //bmp_img *d_img_orig, *d_img_filt;
	  //bmp_img_read(&h_img_orig, "sample_crop.bmp");
 
    //double h_K[3][3] = {0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11}; // mean filter
    //double d_K[3][3];
    //int height = 216;
    //int width = 216;
    // //bmp_img h_img_filt;
    //bmp_img_init_df(&h_img_filt, height, width);
  
    //int SIZE = 32;
    //int nThreads = SIZE;
    
    //int N = 216;
    //int M = 216;
    
    //int PINNED = 0;
    //int numBytesK = sizeof(h_K);
    //int numBytesOrig = sizeof(bmp_img);
    //int numBytesFilt = sizeof(bmp_img);
    /*
     if (PINNED) {
      // Obtiene Memoria [pinned] en el host
      cudaMallocHost((double**)&h_K, numBytesK); 
      cudaMallocHost((bmp_img*)&h_img_orig, numBytesOrig); 
      cudaMallocHost((bmp_img*)&h_img_filt, numBytesFilt); 
    }
    else {
      // Obtener Memoria en el host
      h_K = (double*) malloc(numBytesK); 
      h_img_orig = (bmp_img) malloc(numBytesOrig); 
      h_img_filt = (bmp_img) malloc(numBytesFilt); 
    }
    */
    // Obtener Memoria en el device
    //cudaMalloc((double**)&d_K, numBytesK); 
    //cudaMalloc((&bmp_img)&d_img_orig, numBytesOrig); 
    //cudaMalloc((&bmp_img)&d_img_filt, numBytesFilt);
    
    // Copiar datos desde el host en el device
    //cudaError err = cudaMemcpy(d_K, &h_K, numBytesK, cudaMemcpyHostToDevice);
    //if (err != cudaSuccess) {
    //  printf("1: CUDA error copying to Host: %s\n", cudaGetErrorString(err));
    //}
    //err = cudaMemcpy(&d_img_orig, &h_img_orig, numBytesOrig, cudaMemcpyHostToDevice);
    //if (err!=cudaSuccess) {
    //  printf("2: CUDA error copying to Host: %s\n", cudaGetErrorString(err));
    //}

    // numero de Blocks en cada dimension 
    //int nBlocksN = (N+nThreads-1)/nThreads; 
    //int nBlocksM = (M+nThreads-1)/nThreads; 
    //dim3 dimGrid(nBlocksM, nBlocksN, 1);
    //dim3 dimBlock(nThreads, nThreads, 1);
    //Kernel01<<<dimGrid, dimBlock>>>(3, d_K, d_img_orig, 216, 216, d_img_filt);
    
    //cudaMemcpy(&h_img_filt, &d_img_filt, numBytesFilt, cudaMemcpyDeviceToHost); 

    //Liberar Memoria del device 
    //cudaFree(d_K);
    //cudaFree(&d_img_orig);
    //cudaFree(&d_img_filt);
    
    //bmp_img_write(&h_img_filt, "sample_filtered2.bmp");
    /*
    if (PINNED) {
      cudaFreeHost(h_A); cudaFreeHost(h_img_orig); cudaFreeHost(h_img_filt);
    }
    else {
      free(h_K); free(h_img_orig); free(h_img_filt);
    }*/
    
    //bmp_img_free(&h_img_orig);
    //bmp_img_free(&h_img_filt);
    
    
    // cuda intent 2
    
    int height = 216;
    int width = 216;
    int channels = 3;
    
    unsigned char *h_matrix_orig;
    h_matrix_orig = (unsigned char *) malloc (sizeof(unsigned char)*height*width*channels);
    bmp_img h_img_orig;
    bmp_img_read(&h_img_orig, "sample_crop.bmp");
    img_to_matrix2(h_img_orig, height, width, h_matrix_orig);

    printf("matrix: %d\n", (int)h_matrix_orig[(215*width +215)*3+2]);
    unsigned char *h_matrix_filt;
    h_matrix_filt = (unsigned char *) malloc (sizeof(unsigned char)*height*width*channels);
    double h_K[9] = {0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11};
    unsigned char *d_matrix_orig, *d_matrix_filt;
    double *d_K;
    
    int numBytesK = sizeof(h_K);
    int numBytesOrig = sizeof(char)*height*width*channels;
    int numBytesFilt = sizeof(char)*height*width*channels;
    
    // Obtener Memoria en el device
    cudaMalloc((double**)&d_K, numBytesK); 
    cudaMalloc((unsigned char ***)&d_matrix_orig, numBytesOrig); 
    cudaMalloc((unsigned char ***)&d_matrix_filt, numBytesFilt);
    //cudaMallocPitch(unsigned char ***)&d_matrix_orig, height, width, channels);
    //cudaMallocPitch(unsigned char ***)&d_matrix_filt, height, width, channels);
    
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
    int N = 216;
    int M = 216;
    // numero de Blocks en cada dimension 
    int nBlocksN = (N+nThreads-1)/nThreads; 
    int nBlocksM = (M+nThreads-1)/nThreads; 
    dim3 dimGrid(nBlocksM, nBlocksN, 1);
    dim3 dimBlock(nThreads, nThreads, 1);
    Kernel012<<<dimGrid, dimBlock>>>(3, d_K, d_matrix_orig, 216, 216, d_matrix_filt);
    if (cudaSuccess != cudaGetLastError())
      printf("3: CUDA error at kernel exec: %s\n", cudaGetErrorString(cudaGetLastError()));
    err = cudaMemcpy(h_matrix_filt, d_matrix_filt, numBytesFilt, cudaMemcpyDeviceToHost);
    if (err!=cudaSuccess) {
      printf("4: CUDA error copying to Host: %s\n", cudaGetErrorString(err));
    }
    
    cudaFree(d_K);
    cudaFree(d_matrix_orig);
    cudaFree(d_matrix_filt);
    
    bmp_img h_img_filt;
    bmp_img_init_df(&h_img_filt, height, width);
    matrix_to_img2(h_matrix_filt, height, width, h_img_filt);
    bmp_img_write(&h_img_filt, "sample_filtered2.bmp");
    
    free(h_matrix_orig);
    free(h_matrix_filt);
    
    bmp_img_free(&h_img_orig);
    bmp_img_free(&h_img_filt);

    /*
    int height = 216;
    int width = 216;
    int channels = 3;
    
    unsigned char *h_matrix_orig;
    h_matrix_orig = (unsigned char *) malloc (sizeof(unsigned char)*height*width*channels);
    bmp_img h_img_orig;
    bmp_img_read(&h_img_orig, "sample_crop.bmp");
    img_to_matrix2(h_img_orig, height, width, h_matrix_orig);

    printf("matrix: %d\n", (int)h_matrix_orig[215*width + 215*3 + 2]);
    unsigned char *h_matrix_filt;
    h_matrix_filt = (unsigned char *) malloc (sizeof(unsigned char)*height*width*channels);
    double h_K[9] = {0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11};
    unsigned char *d_matrix_orig, *d_matrix_filt;
    double *d_K;
    
    int numBytesK = sizeof(h_K);
    int numBytesOrig = sizeof(char)*height*width*channels;
    int numBytesFilt = sizeof(char)*height*width*channels;
    
    // Obtener Memoria en el device
    //cudaMalloc((double*)&d_K, numBytesK); 
    //cudaMalloc((unsigned char *)&d_matrix_orig, numBytesOrig); 
    //cudaMalloc((unsigned char *)&d_matrix_filt, numBytesFilt);
    //cudaMallocPitch(unsigned char ***)&d_matrix_orig, height, width, channels);
    //cudaMallocPitch(unsigned char ***)&d_matrix_filt, height, width, channels);
    
    // Copiar datos desde el host en el device
    
    //cudaError err = cudaMemcpy(d_K, h_K, numBytesK, cudaMemcpyHostToDevice);
    //if (err != cudaSuccess) {
    //  printf("1: CUDA error copying to Device: %s\n", cudaGetErrorString(err));
    //}
    //err = cudaMemcpy(d_matrix_orig, h_matrix_orig, numBytesOrig, cudaMemcpyHostToDevice);
    //if (err!=cudaSuccess) {
    //  printf("2: CUDA error copying to Device: %s\n", cudaGetErrorString(err));
    //}

 
    // Ejecutar el kernel 
    int SIZE = 32;
    int nThreads = SIZE;
    int N = 216;
    int M = 216;
    // numero de Blocks en cada dimension 
    int nBlocksN = (N+nThreads-1)/nThreads; 
    int nBlocksM = (M+nThreads-1)/nThreads; 
    //dim3 dimGrid(nBlocksM, nBlocksN, 1);
    //dim3 dimBlock(nThreads, nThreads, 1);
    //Kernel012<<<dimGrid, dimBlock>>>(3, d_K, d_matrix_orig, 216, 216, d_matrix_filt);
    //if (cudaSuccess != cudaGetLastError())
    //  printf("3: CUDA error at kernel exec: %s\n", cudaGetErrorString(cudaGetLastError()));
    //err = cudaMemcpy(h_matrix_filt, d_matrix_filt, numBytesFilt, cudaMemcpyDeviceToHost);
    //if (err!=cudaSuccess) {
    //  printf("4: CUDA error copying to Host: %s\n", cudaGetErrorString(err));
    //}
    
    //cudaFree(d_K);
    //cudaFree(d_matrix_orig);
    //cudaFree(d_matrix_filt);
    seq(3, h_K, h_matrix_orig, 216, 216, h_matrix_filt);
    
    bmp_img h_img_filt;
    bmp_img_init_df(&h_img_filt, height, width);
    matrix_to_img2(h_matrix_filt, height, width, h_img_filt);
    bmp_img_write(&h_img_filt, "sample_filtered2.bmp");
    
    free(h_matrix_orig);
    free(h_matrix_filt);
    
    bmp_img_free(&h_img_orig);
    bmp_img_free(&h_img_filt);
    */
    
    
    
    
    return 0;

}