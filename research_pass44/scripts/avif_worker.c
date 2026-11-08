#include <avif/avif.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

static double now_s(void) { LARGE_INTEGER c,f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f); return (double)c.QuadPart/(double)f.QuadPart; }
static unsigned char *read_file(const char *p, size_t *n) { FILE *f=fopen(p,"rb"); if(!f)return NULL; fseek(f,0,SEEK_END); long z=ftell(f); rewind(f); if(z<0){fclose(f);return NULL;} unsigned char *b=(unsigned char*)malloc((size_t)z); if(!b){fclose(f);return NULL;} if(fread(b,1,(size_t)z,f)!=(size_t)z){free(b);fclose(f);return NULL;} fclose(f); *n=(size_t)z; return b; }
static int write_file(const char *p,const void *b,size_t n){FILE *f=fopen(p,"wb");if(!f)return 0;int ok=fwrite(b,1,n,f)==n;fclose(f);return ok;}
int main(int argc,char **argv){
 if(argc<2)return 2;
 if(!strcmp(argv[1],"enc")){
  if(argc!=7)return 2; int w=atoi(argv[4]),h=atoi(argv[5]),q=atoi(argv[6]); size_t n=(size_t)w*h*3; size_t rn=0; unsigned char *pixels=read_file(argv[2],&rn); if(!pixels||rn!=n)return 3;
  fprintf(stderr,"loaded %zu\\n",rn); avifImage *im=avifImageCreate((uint32_t)w,(uint32_t)h,8,AVIF_PIXEL_FORMAT_YUV420); if(!im)return 4; im->yuvRange=AVIF_RANGE_FULL; im->colorPrimaries=(avifColorPrimaries)1; im->transferCharacteristics=(avifTransferCharacteristics)13; im->matrixCoefficients=(avifMatrixCoefficients)1;
  fprintf(stderr,"image\\n"); avifRGBImage rgb; avifRGBImageSetDefaults(&rgb,im); rgb.format=AVIF_RGB_FORMAT_RGB; rgb.pixels=pixels; rgb.rowBytes=(uint32_t)w*3;
  fprintf(stderr,"converted setup\\n"); avifEncoder *enc=avifEncoderCreate(); if(!enc)return 5; enc->codecChoice=AVIF_CODEC_CHOICE_AOM; enc->maxThreads=1; enc->speed=8; enc->quality=q; enc->qualityAlpha=0;
  fprintf(stderr,"encoder\\n"); double t=now_s(); avifResult r=avifImageRGBToYUV(im,&rgb); if(r==AVIF_RESULT_OK)r=avifEncoderAddImage(enc,im,1,AVIF_ADD_IMAGE_FLAG_SINGLE); avifRWData data=AVIF_DATA_EMPTY; if(r==AVIF_RESULT_OK)r=avifEncoderFinish(enc,&data); double elapsed=now_s()-t; fprintf(stderr,"encoded result %d\\n",r);
  if(r!=AVIF_RESULT_OK){fprintf(stderr,"encode: %s\n",avifResultToString(r));return 6;} if(!write_file(argv[3],data.data,data.size))return 7; printf("%.9f %zu\n",elapsed,data.size); avifRWDataFree(&data); avifEncoderDestroy(enc); avifImageDestroy(im); free(pixels); return 0;
 }
 if(!strcmp(argv[1],"dec")){
  if(argc!=5)return 2; int w=atoi(argv[3]),h=atoi(argv[4]); size_t n=0; unsigned char *data=read_file(argv[2],&n); if(!data)return 3; avifDecoder *dec=avifDecoderCreate(); fprintf(stderr,"loaded %zu\\n",n); avifImage *im=avifImageCreateEmpty(); if(!dec||!im)return 4; avifRGBImage rgb; memset(&rgb,0,sizeof(rgb));
  double t=now_s(); avifResult r=avifDecoderReadMemory(dec,im,data,n); if(r==AVIF_RESULT_OK){avifRGBImageSetDefaults(&rgb,im);rgb.format=AVIF_RGB_FORMAT_RGB;rgb.maxThreads=1;r=avifRGBImageAllocatePixels(&rgb);if(r==AVIF_RESULT_OK)r=avifImageYUVToRGB(im,&rgb);} double elapsed=now_s()-t; fprintf(stderr,"encoded result %d\\n",r);
  if(r!=AVIF_RESULT_OK){fprintf(stderr,"decode: %s\n",avifResultToString(r));return 5;} if(rgb.width!=(uint32_t)w||rgb.height!=(uint32_t)h)return 6; if(!write_file("",rgb.pixels,0)){} /* output is written by caller via temporary below */
  if(argc==5){char path[4096];snprintf(path,sizeof(path),"%s.rgb",argv[2]);if(!write_file(path,rgb.pixels,(size_t)w*h*3))return 7;}
  printf("%.9f %zu %u %u\n",elapsed,(size_t)w*h*3,rgb.width,rgb.height); avifRGBImageFreePixels(&rgb);avifImageDestroy(im);avifDecoderDestroy(dec);free(data);return 0;
 }
 return 2;
}



