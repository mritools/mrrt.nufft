#include "nufft_table.h"

#ifdef TYPE
#error TYPE should not be defined here.
#else

#define TYPE float
#include "nufft_table.template.c"
#undef TYPE

#define TYPE double
#include "nufft_table.template.c"
#undef TYPE

#endif /* TYPE */
