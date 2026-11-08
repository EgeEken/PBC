#include <avif/avif.h>
#include "avifpng.h"
#include "avifjpeg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static avifImage *read_image(const char *path) {
  avifDecoder *decoder = avifDecoderCreate();
  avifImage *image = avifImageCreateEmpty();
  if (!decoder || !image) return NULL;
  decoder->maxThreads = 1;
  avifResult result = avifDecoderReadFile(decoder, image, path);
  if (result != AVIF_RESULT_OK) {
    fprintf(stderr, "%s: %s\n", path, avifResultToString(result));
    avifImageDestroy(image); image = NULL;
  }
  avifDecoderDestroy(decoder);
  return image;
}

static int write_plane(const avifImage *image, int channel, const char *path) {
  const uint32_t width = avifImagePlaneWidth(image, channel);
  const uint32_t height = avifImagePlaneHeight(image, channel);
  const size_t bps = image->depth > 8 ? 2 : 1;
  FILE *file = fopen(path, "wb");
  if (!file) return 0;
  for (uint32_t y = 0; y < height; ++y) {
    const uint8_t *src = image->yuvPlanes[channel] + y * image->yuvRowBytes[channel];
    if (fwrite(src, bps, width, file) != width) { fclose(file); return 0; }
  }
  return fclose(file) == 0;
}

static int load_plane(avifImage *image, int channel, const char *path) {
  const uint32_t width = avifImagePlaneWidth(image, channel);
  const uint32_t height = avifImagePlaneHeight(image, channel);
  const size_t bps = image->depth > 8 ? 2 : 1;
  FILE *file = fopen(path, "rb");
  if (!file) return 0;
  for (uint32_t y = 0; y < height; ++y) {
    uint8_t *dst = image->yuvPlanes[channel] + y * image->yuvRowBytes[channel];
    if (fread(dst, bps, width, file) != width) { fclose(file); return 0; }
  }
  const int ok = fgetc(file) == EOF;
  fclose(file);
  return ok;
}

static int copy_plane(avifImage *dst, const avifImage *src, int channel) {
  const uint32_t width = avifImagePlaneWidth(dst, channel);
  const uint32_t height = avifImagePlaneHeight(dst, channel);
  const size_t bps = dst->depth > 8 ? 2 : 1;
  if (width != avifImagePlaneWidth(src, channel) || height != avifImagePlaneHeight(src, channel)) return 0;
  for (uint32_t y = 0; y < height; ++y) {
    memcpy(dst->yuvPlanes[channel] + y * dst->yuvRowBytes[channel],
           src->yuvPlanes[channel] + y * src->yuvRowBytes[channel], width * bps);
  }
  return 1;
}

static int write_rgb(const avifImage *image, const char *path) {
  avifRGBImage rgb;
  avifRGBImageSetDefaults(&rgb, image);
  rgb.depth = 8;
  rgb.format = AVIF_RGB_FORMAT_RGB;
  rgb.chromaUpsampling = AVIF_CHROMA_UPSAMPLING_AUTOMATIC;
  if (avifRGBImageAllocatePixels(&rgb) != AVIF_RESULT_OK) return 0;
  if (avifImageYUVToRGB(image, &rgb) != AVIF_RESULT_OK) { avifRGBImageFreePixels(&rgb); return 0; }
  FILE *file = fopen(path, "wb");
  if (!file) { avifRGBImageFreePixels(&rgb); return 0; }
  for (uint32_t y = 0; y < rgb.height; ++y) {
    if (fwrite(rgb.pixels + y * rgb.rowBytes, 3, rgb.width, file) != rgb.width) {
      fclose(file); avifRGBImageFreePixels(&rgb); return 0;
    }
  }
  const int ok = fclose(file) == 0;
  avifRGBImageFreePixels(&rgb);
  return ok;
}

static avifImage *clone_image(const avifImage *src) {
  avifImage *copy = avifImageCreateEmpty();
  if (!copy || avifImageCopy(copy, src, AVIF_PLANES_ALL) != AVIF_RESULT_OK) {
    avifImageDestroy(copy); return NULL;
  }
  return copy;
}

static void paths(const char *prefix, const char *kind, const char *plane, char *out, size_t size) {
  snprintf(out, size, "%s_%s_%s.bin", prefix, kind, plane);
}

static int write_meta(const avifImage *image, const char *prefix) {
  char path[4096];
  snprintf(path, sizeof(path), "%s.meta", prefix);
  FILE *file = fopen(path, "w");
  if (!file) return 0;
  fprintf(file, "width=%u\nheight=%u\ndepth=%u\nyuv_format=%d\nyuv_range=%d\nchroma_position=%d\nprimaries=%u\ntransfer=%u\nmatrix=%u\n",
          image->width, image->height, image->depth, image->yuvFormat, image->yuvRange,
          image->yuvChromaSamplePosition, image->colorPrimaries,
          image->transferCharacteristics, image->matrixCoefficients);
  return fclose(file) == 0;
}

static int save_planes(const avifImage *image, const char *prefix, const char *kind) {
  const int channels[] = { AVIF_CHAN_Y, AVIF_CHAN_U, AVIF_CHAN_V };
  const char *labels[] = { "Y", "U", "V" };
  char path[4096];
  for (int i = 0; i < 3; ++i) {
    paths(prefix, kind, labels[i], path, sizeof(path));
    if (!write_plane(image, channels[i], path)) return 0;
  }
  return 1;
}

static int make_source(const char *sourcePath, const char *prefix) {
  avifImage *image = avifImageCreateEmpty();
  if (!image) return 0;
  image->colorPrimaries = AVIF_COLOR_PRIMARIES_BT709;
  image->transferCharacteristics = AVIF_TRANSFER_CHARACTERISTICS_SRGB;
  image->matrixCoefficients = AVIF_MATRIX_COEFFICIENTS_BT709;
  image->yuvRange = AVIF_RANGE_FULL;
  const char *ext = strrchr(sourcePath, '.');
  avifBool ok = AVIF_FALSE;
  if (ext && (!strcmp(ext, ".png") || !strcmp(ext, ".PNG"))) {
    uint32_t sourceDepth = 0;
    ok = avifPNGRead(sourcePath, image, AVIF_PIXEL_FORMAT_YUV420, 8,
                     AVIF_CHROMA_DOWNSAMPLING_AUTOMATIC, AVIF_TRUE, AVIF_TRUE,
                     AVIF_TRUE, AVIF_FALSE, UINT32_MAX, &sourceDepth);
  } else {
    ok = avifJPEGRead(sourcePath, image, AVIF_PIXEL_FORMAT_YUV420, 8,
                      AVIF_CHROMA_DOWNSAMPLING_AUTOMATIC, AVIF_TRUE, AVIF_TRUE,
                      AVIF_TRUE, AVIF_TRUE, UINT32_MAX);
  }
  if (!ok || image->depth != 8 || image->yuvFormat != AVIF_PIXEL_FORMAT_YUV420 ||
      image->yuvRange != AVIF_RANGE_FULL || image->colorPrimaries != AVIF_COLOR_PRIMARIES_BT709 ||
      image->transferCharacteristics != AVIF_TRANSFER_CHARACTERISTICS_SRGB ||
      image->matrixCoefficients != AVIF_MATRIX_COEFFICIENTS_BT709) {
    fprintf(stderr, "pinned source loader returned unexpected image properties\n");
    avifImageDestroy(image); return 0;
  }
  const int result = save_planes(image, prefix, "ref") && write_meta(image, prefix);
  avifImageDestroy(image);
  return result;
}

static int compare(const char *lossyPath, const char *prefix) {
  avifImage *decoded = read_image(lossyPath);
  if (!decoded) return 0;
  avifImage *reference = clone_image(decoded);
  if (!reference) { avifImageDestroy(decoded); return 0; }
  const int channels[] = { AVIF_CHAN_Y, AVIF_CHAN_U, AVIF_CHAN_V };
  const char *labels[] = { "Y", "U", "V" };
  char path[4096];
  for (int i = 0; i < 3; ++i) {
    paths(prefix, "ref", labels[i], path, sizeof(path));
    if (!load_plane(reference, channels[i], path)) return 0;
  }
  if (!save_planes(decoded, prefix, "decoded") || !write_meta(decoded, prefix)) return 0;
  snprintf(path, sizeof(path), "%s_reference_rgb.bin", prefix);
  if (!write_rgb(reference, path)) return 0;
  snprintf(path, sizeof(path), "%s_decoded_rgb.bin", prefix);
  if (!write_rgb(decoded, path)) return 0;
  avifImage *decodedY = clone_image(reference);
  avifImage *decodedUV = clone_image(reference);
  if (!decodedY || !decodedUV || !copy_plane(decodedY, decoded, AVIF_CHAN_Y) ||
      !copy_plane(decodedUV, decoded, AVIF_CHAN_U) || !copy_plane(decodedUV, decoded, AVIF_CHAN_V)) return 0;
  snprintf(path, sizeof(path), "%s_decodedY_referenceUV_rgb.bin", prefix);
  if (!write_rgb(decodedY, path)) return 0;
  snprintf(path, sizeof(path), "%s_referenceY_decodedUV_rgb.bin", prefix);
  if (!write_rgb(decodedUV, path)) return 0;
  avifImageDestroy(decodedY); avifImageDestroy(decodedUV);
  avifImageDestroy(decoded); avifImageDestroy(reference);
  return 1;
}

int main(int argc, char **argv) {
  if (argc == 4 && !strcmp(argv[1], "source")) return make_source(argv[2], argv[3]) ? 0 : 1;
  if (argc == 4 && !strcmp(argv[1], "compare")) return compare(argv[2], argv[3]) ? 0 : 1;
  fprintf(stderr, "usage: plane_probe source input_image prefix | compare lossy.avif prefix\n");
  return 2;
}
