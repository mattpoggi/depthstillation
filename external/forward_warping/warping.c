#include <stdlib.h>
#include <stdio.h>
#define valid(X, Y, W)  (Y*W*5+X*5+3)
#define collision(X, Y, W)  (Y*W*5+X*5+4)

void forward_warping(const void *src, const void *idx, const void *idy, const void *z, void *warped, int h, int w)
{
    float *dlut = (float *)calloc(h * w, sizeof(float));
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            dlut[i * w + j] = 1000;

    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        {
            int x = ((long *)idx)[i * w + j];
            int y = ((long *)idy)[i * w + j];
            
            if (((float *)z)[i * w + j] < dlut[y * w + x])
                for (int c = 0; c < 3; c++)
                    ((unsigned char *)warped)[y * w * 5 + x * 5 + c] = ((unsigned char *)src)[i * w * 3 + j * 3 + c];

            ((unsigned char *)warped)[valid(x,y,w)] = 1;            	    
            if (dlut[y * w + x] != 1000)
                ((unsigned char *)warped)[collision(x,y,w)] = 0;
            else
                ((unsigned char *)warped)[collision(x,y,w)] = 1;            	    
            dlut[y * w + x] = ((float *)z)[i * w + j];
        }

    free(dlut);
    return;
}
