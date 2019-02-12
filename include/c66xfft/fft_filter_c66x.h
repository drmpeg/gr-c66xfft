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

#ifndef INCLUDED_C66XFFT_FFT_FILTER_C66X_H
#define INCLUDED_C66XFFT_FFT_FILTER_C66X_H

#include <c66xfft/api.h>
#include <vector>
#include <gnuradio/gr_complex.h>
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
    namespace kernel {

      /*!
       * \brief Fast FFT filter with gr_complex input, gr_complex output and gr_complex taps
       * \ingroup filter_blk
       *
       * \details
       * This block performs fast convolution using the
       * overlap-and-save algorithm. The filtering is performand in
       * the frequency domain instead of the time domain (see
       * gr::filter::kernel::fir_filter_ccc). For an input signal x
       * and filter coefficients (taps) t, we compute y as:
       *
       * \code
       *    y = ifft(fft(x)*fft(t))
       * \endcode
       *
       * This kernel computes the FFT of the taps when they are set to
       * only perform this operation once. The FFT of the input signal
       * x is done every time.
       *
       * Because this is designed as a very low-level kernel
       * operation, it is designed for speed and avoids certain checks
       * in the filter() function itself. The filter function expects
       * that the input signal is a multiple of d_nsamples in the
       * class that's computed internally to be as fast as
       * possible. The function set_taps will return the value of
       * nsamples that can be used externally to check this
       * boundary. Notice that all implementations of the fft_filter
       * GNU Radio blocks (e.g., gr::c66xfft::fft_filter_ccc_c66x) use this
       * value of nsamples to compute the value to call
       * gr::block::set_output_multiple that ensures the scheduler
       * always passes this block the right number of samples.
       */
      class C66XFFT_API fft_filter_ccc_c66x
      {
       private:
        int                          d_ntaps;
        int                          d_nsamples;
        int                          d_fftsize;         // fftsize = ntaps + nsamples - 1
        int                          d_decimation;
        std::vector<gr_complex>      d_tail;            // state carried between blocks for overlap-add
        std::vector<gr_complex>      d_taps;            // stores time domain taps
        gr_complex                   *d_xformed_taps;   // Fourier xformed taps
        gr_complex                   *x;
        gr_complex                   *y;
        float                        *w;
        float                        *z;
        int                          NUMCOMPUNITS;
        int                          FFTRADIX;
        cl::Context                  context;
        std::vector<cl::Device>      devices;
        cl::CommandQueue             Q;
        cl::Buffer                   bufX;
        cl::Buffer                   bufY;
        cl::Buffer                   bufW;
        cl::Buffer                   bufZ;
        cl::Program::Binaries        binary;
        cl::Program                  program;
        cl::Kernel                   kernel;
        cl::KernelFunctor            null;
        cl::Kernel                   fwdfft;
        cl::Kernel                   invfft;

        void compute_sizes(int ntaps);
        int tailsize() const { return d_ntaps - 1; }
        void tw_gen_forward(float *w, int n);
        void tw_gen_reverse(float *w, int n);

       public:
        /*!
         * \brief Construct an FFT filter for complex vectors with the given taps and decimation rate.
         *
         * This is the basic implementation for performing FFT filter for fast convolution
         * in other blocks (e.g., gr::c66xfft::fft_filter_ccc_c66x).
         *
         * \param decimation The decimation rate of the filter (int)
         * \param taps       The filter taps (vector of complex)
         */
        fft_filter_ccc_c66x(int decimation,
                       const std::vector<gr_complex> &taps);

        ~fft_filter_ccc_c66x();

        /*!
         * \brief Set new taps for the filter.
         *
         * Sets new taps and resets the class properties to handle different sizes
         * \param taps       The filter taps (complex)
         */
        int set_taps(const std::vector<gr_complex> &taps);

        /*!
         * \brief Returns the taps.
         */
        std::vector<gr_complex> taps() const;

        /*!
         * \brief Returns the number of taps in the filter.
         */
        unsigned int ntaps() const;

        /*!
         * \brief Perform the filter operation
         *
         * \param nitems  The number of items to produce
         * \param input   The input vector to be filtered
         * \param output  The result of the filter operation
         */
        int filter(int nitems, const gr_complex *input, gr_complex *output);
      };

    } /* namespace kernel */
  } /* namespace c66xfft */
} /* namespace gr */

#endif /* INCLUDED_C66XFFT_FFT_FILTER_C66X_H */
