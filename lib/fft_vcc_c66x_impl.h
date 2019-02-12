/* -*- c++ -*- */
/* 
 * Copyright 2019 Ron Economos.
 * 
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_C66XFFT_FFT_VCC_C66X_IMPL_H
#define INCLUDED_C66XFFT_FFT_VCC_C66X_IMPL_H

#include <c66xfft/fft_vcc_c66x.h>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include "ocl_util.h"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/TI/cl.hpp>

namespace gr {
  namespace c66xfft {

    class fft_vcc_c66x_impl : public fft_vcc_c66x
    {
     private:
      gr_complex                *x;
      gr_complex                *y;
      float                     *w;
      unsigned int              d_fft_size;
      std::vector<float>        d_window;
      bool                      d_forward;
      bool                      d_shift;
      int                       NUMCOMPUNITS;
      int                       FFTRADIX;
      cl::Context               context;
      std::vector<cl::Device>   devices;
      cl::CommandQueue          Q;
      cl::Buffer                bufX;
      cl::Buffer                bufY;
      cl::Buffer                bufW;
      cl::Program::Binaries     binary;
      cl::Program               program;
      cl::Kernel                kernel;
      cl::KernelFunctor         null;
      cl::Kernel                fft;

     public:
      fft_vcc_c66x_impl(int fft_size, bool forward, const std::vector<float> &window, bool shift);
      ~fft_vcc_c66x_impl();

      bool set_window(const std::vector<float> &window);
      void tw_gen_forward(float *w, int n);
      void tw_gen_reverse(float *w, int n);

      int work(int noutput_items,
         gr_vector_const_void_star &input_items,
         gr_vector_void_star &output_items);
    };

  } // namespace c66xfft
} // namespace gr

#endif /* INCLUDED_C66XFFT_FFT_VCC_C66X_IMPL_H */

