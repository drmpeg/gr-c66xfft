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

#ifndef INCLUDED_C66XFFT_FFT_FILTER_CCC_C66X_IMPL_H
#define INCLUDED_C66XFFT_FFT_FILTER_CCC_C66X_IMPL_H

#include <c66xfft/fft_filter_c66x.h>
#include <c66xfft/fft_filter_ccc_c66x.h>

namespace gr {
  namespace c66xfft {

    class fft_filter_ccc_c66x_impl : public fft_filter_ccc_c66x
    {
     private:
      int d_nsamples;
      bool d_updated;
      kernel::fft_filter_ccc_c66x *d_filter;
      std::vector<gr_complex> d_new_taps;

     public:
      fft_filter_ccc_c66x_impl(int decimation, const std::vector<gr_complex> &taps);
      ~fft_filter_ccc_c66x_impl();

      void set_taps(const std::vector<gr_complex> &taps);
      std::vector<gr_complex> taps() const;

      int work(int noutput_items,
         gr_vector_const_void_star &input_items,
         gr_vector_void_star &output_items);
    };

  } // namespace c66xfft
} // namespace gr

#endif /* INCLUDED_C66XFFT_FFT_FILTER_CCC_C66X_IMPL_H */

