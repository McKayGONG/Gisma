#pragma once
#include <time.h>

#ifndef TIME_UTC
#  define TIME_UTC 1
#endif

#ifdef __cplusplus
extern "C" {
#endif
static inline int timespec_get(struct timespec* ts, int base)
{
    if (base != TIME_UTC) return 0;
    return clock_gettime(CLOCK_REALTIME, ts) == 0 ? base : 0;
}
#ifdef __cplusplus
}
#endif
