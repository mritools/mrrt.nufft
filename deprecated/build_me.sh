#g++-4.8 -Wall -O3 -shared -fPIC ./interp_table.cpp -o interp_table_lib.so -fopenmp
g++-4.8 -Wall -O3 -shared -fPIC ./interp_table.cpp -o libinterp_table_lib.so -fopenmp -v  #  -ftls-model=local-dynamic

# readelf -a -W ./libinterp_table_lib.so  | grep -i TLS
# /usr/lib/x86_64-linux-gnu/libgomp.so.1

#NOTE: if gcc-4.6 is used will get errors about undefined symbol: __gxx_personality_v0

#gcc -pthread -fno-strict-aliasing -g -O2 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/home/lee8rx/anaconda/lib/python2.7/site-packages/numpy/core/include -Isrc -I/home/lee8rx/anaconda/include/python2.7 -c denspeed.c -o denspeed.o -fopenmp
#gcc -pthread -shared denspeed.o -L/home/lee8rx/anaconda/lib -lpython2.7 -o denspeed.so -fopenmp


#NOTE:  SOME ISSUE WITH TLS ERRORS.  MAY BE BUG IN libgomp (gcc's OpenMP library)
# see:  https://sourceware.org/ml/libc-alpha/2015-02/msg00315.html

# diff -Naur ./libgomp/configure.tgt ../../gcc-4.2.0/libgomp/configure.tgt
# --- ./libgomp/configure.tgt 2006-12-02 18:02:00.000000000 -0200
# +++ ../../gcc-4.2.0/libgomp/configure.tgt   2007-07-07 15:24:51.000000000 -0300
# @@ -17,8 +17,8 @@
#    case "${target}" in

#      *-*-linux*)
# -   XCFLAGS="${XCFLAGS} -ftls-model=initial-exec"
# -   XLDFLAGS="${XLDFLAGS} -Wl,-z,nodlopen"
# +#  XCFLAGS="${XCFLAGS} -ftls-model=initial-exec"
# +#  XLDFLAGS="${XLDFLAGS} -Wl,-z,nodlopen"
#     ;;
#    esac
#  fi