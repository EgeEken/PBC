#include <stdint.h>
#include <stddef.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

API int mse_u8(const uint8_t *a, const uint8_t *b, int64_t h, int64_t w, int64_t c,
               const int64_t *sa, const int64_t *sb, double *out) {
    if (!a || !b || !sa || !sb || !out || h<=0 || w<=0 || c<=0 || c>4 ||
        h>32000000 || w>32000000 || h>32000000/w) return 1;
    uint64_t total=0;
    if (sa[2]==1 && sb[2]==1 && sa[1]==c && sb[1]==c && sa[0]==w*c && sb[0]==w*c) {
        int64_t n=h*w*c;
        for (int64_t i=0;i<n;i++) {
            int d=(int)a[i]-(int)b[i];
            total+=(uint64_t)(d*d);
        }
    } else {
        for (int64_t y=0;y<h;y++) for (int64_t x=0;x<w;x++) for (int64_t k=0;k<c;k++) {
            int d=(int)a[y*sa[0]+x*sa[1]+k*sa[2]]-(int)b[y*sb[0]+x*sb[1]+k*sb[2]];
            total+=(uint64_t)(d*d);
        }
    }
    *out=(double)total/(double)(h*w*c);
    return 0;
}
