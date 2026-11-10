// cv2.resize(INTER_CUBIC) of an 8-bit picture as OpenCV 5.0.0's ipp_resize (imgproc/src/resize.cpp)
// runs it: the Integration Wrappers' resize with OpenCV's parameters (IwiResizeParams(0, 0, 0.75,
// 4): no antialiasing, cubic B 0 and C 0.75), the border replicated, over the output rows of one of
// its parallel tiles; IPP initialised as OpenCV's IPPInitSingleton does it (ippInit: the CPU's best
// code, k0 on these AVX-512 boxes). ippicv 2026.0.0, the build OpenCV 5.0.0 bundles.
#include <stdio.h>
#include "iw/iw_image_transform.h"

static int initialised = 0;

static void init(void) {
    if (!initialised) {
        ippInit();
        initialised = 1;
    }
}

// the image library's name (with the code it dispatched to), version and build date, as
// cv2.ipp.getIppVersion() prints them
const char *rustyui_ipp_version(void) {
    static char buf[256];
    init();
    const IppLibraryVersion *v = ippiGetLibVersion();
    if (!v)
        return "";
    snprintf(buf, sizeof buf, "%s %s %s", v->Name, v->Version, v->BuildDate);
    return buf;
}

int rustyui_ipp_resize_cubic_8u(const unsigned char *src, long long src_step, int sw, int sh, unsigned char *dst,
                                long long dst_step, int dw, int dh, int channels, int row0, int row1) {
    init();
    IwiImage s, d;
    iwiImage_Init(&s);
    iwiImage_Init(&d);
    IwiSize ss = {sw, sh}, ds = {dw, dh};
    IwiBorderSize none = {0, 0, 0, 0};
    IppStatus st = iwiImage_InitExternalConst(&s, ss, ipp8u, channels, &none, src, src_step);
    if (st < 0)
        return st;
    st = iwiImage_InitExternal(&d, ds, ipp8u, channels, &none, dst, dst_step);
    if (st < 0)
        return st;
    IwiResizeParams p;
    iwiResize_SetDefaultParams(&p);
    p.antialiasing = 0;
    p.cubicBVal = 0;
    p.cubicCVal = 0.75f;
    p.lanczosLobes = 4;
    IwiResizeSpec *spec = NULL;
    st = iwiResize_InitAlloc(&spec, ss, ds, ipp8u, channels, ippCubic, &p, ippBorderRepl);
    if (st < 0)
        return st;
    IwiRoi roi = {0, row0, dw, row1 - row0};
    IwiTile tile = iwiTile_SetRoi(roi);
    st = iwiResize_Process(spec, &s, &d, ippBorderRepl, NULL, &tile);
    iwiResize_Free(spec);
    return st;
}
