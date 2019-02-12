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


#ifndef INCLUDED_C66XFFT_FFT_VCC_C66X_H
#define INCLUDED_C66XFFT_FFT_VCC_C66X_H

#include <c66xfft/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
  namespace c66xfft {

    /*!
     * \brief <+description of block+>
     * \ingroup c66xfft
     *
     */
    class C66XFFT_API fft_vcc_c66x : virtual public gr::sync_block
    {
     public:
      typedef boost::shared_ptr<fft_vcc_c66x> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of c66xfft::fft_vcc_c66x.
       *
       * To avoid accidental use of raw pointers, c66xfft::fft_vcc_c66x's
       * constructor is in a private implementation
       * class. c66xfft::fft_vcc_c66x::make is the public interface for
       * creating new instances.
       */
      static sptr make(int fft_size, bool forward, const std::vector<float> &window, bool shift);
    };

  } // namespace c66xfft
} // namespace gr

#endif /* INCLUDED_C66XFFT_FFT_VCC_C66X_H */

