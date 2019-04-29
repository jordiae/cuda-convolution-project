#include <stdio.h>
#include "lib/libbmp.h"

// Sequential version of convolution (should be refactored into a function).
// The code *is* ours, but based on Wikipedia's pseudo-code for convolution
// https://en.wikipedia.org/wiki/Kernel_(image_processing)#Convolution
// The only difference is that we are doing it for the 3 channel/colours.
//double K[3][3] = {1/9}; // mean filter
bmp_img FilterSeq (int size_filter, double FilterMatrix[][size_filter], bmp_img img_orig, int height, int width) {
    bmp_img img_filt;
    bmp_img_init_df(&img_filt, height, width);
    for (size_t i = 0; i < height; i++) { // image row
        for (size_t j = 0; j < width; j++) { // pixels in image row
            double accumulator_red = 0;
            double accumulator_green = 0;
            double accumulator_blue = 0;
            for (size_t k = 0; k < 3; k++) { // kernel rows
                for (size_t l = 0; l < 3; l++) { // kernel elements/cols
                    if ((i % 3 == k) && (j % 3 == l)) {// corresponding element
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




int main (int argc, char *argv[])
{
    bmp_img img_orig;
	bmp_img_read(&img_orig, "sample_crop.bmp");
    double K[3][3] = {0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11}; // mean filter
    bmp_img img_filt  = FilterSeq (3,K, img_orig, 216, 216);
    bmp_img_write(&img_filt, "sample_filtered.bmp");
    bmp_img_free(&img_orig);
    bmp_img_free(&img_filt);
}

