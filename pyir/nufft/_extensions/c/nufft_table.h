/*
The underlying C code for these routines are adapted from C code originally
developed by Jeff Fessler and his students at the University of Michigan.

OpenMP support and the Cython wrappers were created by Gregory R. Lee
(Cincinnati Childrens Hospital Medical Center).

Note:  For simplicity the adjoint NUFFT is only parallelized across multiple
coils and/or repetitions.  This was done for simplicity to avoid any potential
thread conflicts.

The C templating used here is based on the implementation by Kai Wohlfahrt as
developed for the BSD-licensed PyWavelets project.
*/
#pragma once

//#include "common.h"

#ifdef TYPE
#error TYPE should not be defined here.
#else

#define TYPE float
#include "nufft_table.template.h"
#undef TYPE

#define TYPE double
#include "nufft_table.template.h"
#undef TYPE

#endif /* TYPE */
