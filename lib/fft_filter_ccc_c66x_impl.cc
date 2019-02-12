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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include "fft_filter_ccc_c66x_impl.h"

namespace gr {
  namespace c66xfft {

    fft_filter_ccc_c66x::sptr
    fft_filter_ccc_c66x::make(int decimation, const std::vector<gr_complex> &taps)
    {
      return gnuradio::get_initial_sptr
        (new fft_filter_ccc_c66x_impl(decimation, taps));
    }

    /*
     * The private constructor
     */
    fft_filter_ccc_c66x_impl::fft_filter_ccc_c66x_impl(int decimation, const std::vector<gr_complex> &taps)
      : gr::sync_decimator("fft_filter_ccc_c66x",
              gr::io_signature::make (1, 1, sizeof(gr_complex)),
              gr::io_signature::make (1, 1, sizeof(gr_complex)),
              decimation),
              d_updated(false)
    {
      set_history(1);

      d_filter = new kernel::fft_filter_ccc_c66x(decimation, taps);

      d_new_taps = taps;
      d_nsamples = d_filter->set_taps(taps);
      set_output_multiple(d_nsamples);
    }

    /*
     * Our virtual destructor.
     */
    fft_filter_ccc_c66x_impl::~fft_filter_ccc_c66x_impl()
    {
      delete d_filter;
    }

    void
    fft_filter_ccc_c66x_impl::set_taps(const std::vector<gr_complex> &taps)
    {
      d_new_taps = taps;
      d_updated = true;
    }

    std::vector<gr_complex>
    fft_filter_ccc_c66x_impl::taps() const
    {
      return d_new_taps;
    }

    int
    fft_filter_ccc_c66x_impl::work(int noutput_items,
        gr_vector_const_void_star &input_items,
        gr_vector_void_star &output_items)
    {
      const gr_complex *in = (const gr_complex *) input_items[0];
      gr_complex *out = (gr_complex *) output_items[0];

      if (d_updated){
        d_nsamples = d_filter->set_taps(d_new_taps);
        d_updated = false;
        set_output_multiple(d_nsamples);
        return 0;  // output multiple may have changed
      }
      d_filter->filter(noutput_items, in, out);

      return noutput_items;
    }

  } /* namespace c66xfft */
} /* namespace gr */

