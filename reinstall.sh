#!/bin/bash
pip uninstall pyir.nufft -y
LDFLAGS="$LDFLAGS -Wl,-rpath,/Users/lee8rx/anaconda/lib" pip install -e . -v