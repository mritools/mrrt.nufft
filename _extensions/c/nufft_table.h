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
