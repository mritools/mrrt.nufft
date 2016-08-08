#!/bin/bash
pip uninstall pyir.nufft -y
export LDFLAGS="$LDFLAGS -L/Users/lee8rx/anaconda/lib -Wl,-rpath,/Users/lee8rx/anaconda/lib"
pip install -e . -v

install_name_tool -add_rpath /Users/lee8rx/anaconda/lib /Users/lee8rx/src/my_git/pyir/pyir.nufft/pyir/nufft/_extensions/_nufft_table.cpython-35m-darwin.so
