/* -*- c++ -*- */

#define C66XFFT_API

%include "gnuradio.i"			// the common stuff

//load generated python docstrings
%include "c66xfft_swig_doc.i"

%{
#include "c66xfft/fft_vcc_c66x.h"
#include "c66xfft/fft_filter_ccc_c66x.h"
%}


%include "c66xfft/fft_vcc_c66x.h"
GR_SWIG_BLOCK_MAGIC2(c66xfft, fft_vcc_c66x);
%include "c66xfft/fft_filter_ccc_c66x.h"
GR_SWIG_BLOCK_MAGIC2(c66xfft, fft_filter_ccc_c66x);
