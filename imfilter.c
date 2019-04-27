#include <stdio.h>
#include "lib/libbmp.h"

int main (int argc, char *argv[])
{
        // Testing usage of libBMP
        // LibBMP is NOT made by us (see lib/libmp.h).
        bmp_img img1;
	bmp_img_init_df(&img1, 512, 512);
	// Draw a checkerboard pattern:
	for (size_t y = 0; y < 512; y++)
	{
		for (size_t x = 0; x < 512; x++)
		{
			if ((y % 128 < 64 && x % 128 < 64) ||
			    (y % 128 >= 64 && x % 128 >= 64))
			{
				bmp_pixel_init(&img1.img_pixels[y][x], 250, 250, 250);
			}
			else
			{
				bmp_pixel_init(&img1.img_pixels[y][x], 0, 0, 0);
			}
		}
	}
        bmp_img_write(&img1, "checkerboard.bmp");
	bmp_img_free(&img1);
	bmp_img img2;
	bmp_img_read(&img2, "sample.bmp");
        printf("%d\n", (int) img2.img_pixels[66]->red);
        bmp_img_free(&img2);
        
        bmp_img img_orig;
	bmp_img_read(&img_orig, "sample_crop.bmp");
        bmp_img img_filt;
	bmp_img_init_df(&img_filt, 216, 216);
        
        // Now, sequential version of convolution (should be refactored into a function).
        // The code *is* ours, but based on Wikipedia's pseudo-code for convolution
        // https://en.wikipedia.org/wiki/Kernel_(image_processing)#Convolution
        // The only difference is that we are doing it for the 3 channel/colours.
        //double K[3][3] = {1/9}; // mean filter
        double K[3][3] = {0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11}; // mean filter
        printf("Hola %f\n", K[0][0]);
        for (size_t i = 0; i < 216; i++) { // image row
            for (size_t j = 0; j < 216; j++) { // pixels in image row
                double accumulator_red = 0;
                double accumulator_green = 0;
                double accumulator_blue = 0;
                for (size_t k = 0; k < 3; k++) { // kernel rows
                    for (size_t l = 0; l < 3; l++) { // kernel elements/cols
                        if ((i % 3 == k) && (j % 3 == l)) {// corresponding element
                                //printf("hola\n");
                                accumulator_red += K[k][l]*(double)(img_orig.img_pixels[i][j].red);
                                printf("%f %f %f\n", accumulator_red,K[k][l], (double)(img_orig.img_pixels[i][j].red));
                                accumulator_green += K[k][l]*(double)(img_orig.img_pixels[i][j].green);
                                accumulator_blue += K[k][l]*(double)(img_orig.img_pixels[i][j].blue);
                        }
                    }
                    
                }
                printf("adeu %f\n",accumulator_red);
                img_filt.img_pixels[i][j].red = (int) accumulator_red;
                img_filt.img_pixels[i][j].green = (int) accumulator_green;
                img_filt.img_pixels[i][j].blue = (int) accumulator_blue;
                
            }
        }
        bmp_img_write(&img_filt, "sample_filtered.bmp");
        bmp_img_free(&img_orig);
	bmp_img_free(&img_filt);
	return 0;
}
