#pragma once

//#include "common.h"

#ifdef TYPE
#error TYPE should not be defined here.
#else

#define TYPE float
#define CPLX_TYPE float _Complex
#include "nufft_table.template.h"
#undef TYPE
#undef CPLX_TYPE

#define TYPE double
#define CPLX_TYPE double _Complex
#include "nufft_table.template.h"
#undef TYPE
#undef CPLX_TYPE

#endif /* TYPE */
