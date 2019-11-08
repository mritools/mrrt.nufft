#!/bin/bash
pip uninstall mrrt.nufft -y
rm -rf build
rm -rf mrrt.nufft.egg-info
#export LDFLAGS="$LDFLAGS -L/Users/lee8rx/anaconda/lib -Wl,-rpath,/Users/lee8rx/anaconda/lib"
pip install -e . -v

# install_name_tool -add_rpath /Users/lee8rx/anaconda/lib /Users/lee8rx/src/my_git/mrrt/mrrt.nufft/mrrt/nufft/_extensions/_nufft_table.cpython-35m-darwin.so
