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
#include "fft_vcc_c66x_impl.h"
#include <volk/volk.h>

#include "kernel.dsp_h"

#define FFTCHS (64)

namespace gr {
  namespace c66xfft {

    fft_vcc_c66x::sptr
    fft_vcc_c66x::make(int fft_size, bool forward, const std::vector<float> &window, bool shift)
    {
      return gnuradio::get_initial_sptr
        (new fft_vcc_c66x_impl(fft_size, forward, window, shift));
    }

    /*
     * The private constructor
     */
    fft_vcc_c66x_impl::fft_vcc_c66x_impl(int fft_size, bool forward, const std::vector<float> &window, bool shift)
      : gr::sync_block("fft_vcc_c66x",
              gr::io_signature::make(1, 1, fft_size * sizeof(gr_complex)),
              gr::io_signature::make(1, 1, fft_size * sizeof(gr_complex))),
              d_fft_size(fft_size), d_forward(forward), d_shift(shift)
    {
      if (!set_window(window))
        throw std::runtime_error("fft_vcc_c66: window not the same length as FFT Size\n");

      if (fft_size & (fft_size - 1)) {
        throw std::runtime_error("fft_vcc_c66: FFT Size not a power of 2\n");
      }
      if (fft_size & 0x55555555) {
        FFTRADIX = 4;
      }
      else {
        FFTRADIX = 2;
      }
      int channel_size = fft_size * sizeof(gr_complex);
      // __malloc_ddr() returns 128 bytes aligned memory
      x = (gr_complex *) __malloc_ddr(FFTCHS * channel_size);
      y = (gr_complex *) __malloc_ddr(FFTCHS * channel_size);
      // same twiddle factor for all channels of same size
      w = (float *) __malloc_ddr(channel_size);
      if (x == nullptr || y == nullptr || w == nullptr) {
        if (x != nullptr)  __free_ddr(x);
        if (y != nullptr)  __free_ddr(y);
        if (w != nullptr)  __free_ddr(w);
        throw std::runtime_error("fft_vcc_c66: Cannot allocate DDR memory\n");
      }
      context = cl::Context(CL_DEVICE_TYPE_ACCELERATOR);
      devices = context.getInfo<CL_CONTEXT_DEVICES>();
      Q = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
      cl_ulong LOCALMEMSIZE;
      devices[0].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &NUMCOMPUNITS);
      devices[0].getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &LOCALMEMSIZE);

      bufX = cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
                        channel_size * FFTCHS, x);
      bufY = cl::Buffer(context, CL_MEM_WRITE_ONLY|CL_MEM_USE_HOST_PTR,
                        channel_size * FFTCHS, y);
      bufW = cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
                        channel_size, w);

      binary = cl::Program::Binaries(1, std::make_pair(kernel_dsp_bin,
                                     sizeof(kernel_dsp_bin)));
      program = cl::Program(context, devices, binary);
      program.build(devices);
      kernel = cl::Kernel(program, "null");
      null = kernel.bind(Q, cl::NDRange(1), cl::NDRange(1));
      null().wait();
      if (forward) {
        tw_gen_forward(w, fft_size);  // Generate twiddle factors
        fft = cl::Kernel(program, "ocl_DSPF_sp_fftSPxSP");
      }
      else {
        tw_gen_reverse(w, fft_size);  // Generate twiddle factors
        fft = cl::Kernel(program, "ocl_DSPF_sp_ifftSPxSP");
      }
    }

    /*
     * Our virtual destructor.
     */
    fft_vcc_c66x_impl::~fft_vcc_c66x_impl()
    {
      __free_ddr(w);
      __free_ddr(y);
      __free_ddr(x);
    }

    bool
    fft_vcc_c66x_impl::set_window(const std::vector<float> &window)
    {
      if (window.size() == 0 || window.size() == d_fft_size) {
        d_window = window;
        return true;
      }
      else {
        return false;
      }
    }


    void
    fft_vcc_c66x_impl::tw_gen_forward(float *w, int n)
    {
      int i, j, k;

      for (j = 1, k = 0; j <= n >> 2; j = j << 2)
      {
        for (i = 0; i < n >> 2; i += j)
        {
          w[k]     = (float) sin (2 * M_PI * i / n);
          w[k + 1] = (float) cos (2 * M_PI * i / n);
          w[k + 2] = (float) sin (4 * M_PI * i / n);
          w[k + 3] = (float) cos (4 * M_PI * i / n);
          w[k + 4] = (float) sin (6 * M_PI * i / n);
          w[k + 5] = (float) cos (6 * M_PI * i / n);
          k += 6;
        }
      }
    }

    void
    fft_vcc_c66x_impl::tw_gen_reverse(float *w, int n)
    {
      int i, j, k;

      for (j = 1, k = 0; j <= n >> 2; j = j << 2)
      {
        for (i = 0; i < n >> 2; i += j)
        {
          w[k]     = (float) (-1.0) * sin (2 * M_PI * i / n);
          w[k + 1] = (float) cos (2 * M_PI * i / n);
          w[k + 2] = (float) (-1.0) * sin (4 * M_PI * i / n);
          w[k + 3] = (float) cos (4 * M_PI * i / n);
          w[k + 4] = (float) (-1.0) * sin (6 * M_PI * i / n);
          w[k + 5] = (float) cos (6 * M_PI * i / n);
          k += 6;
        }
      }
    }

    int
    fft_vcc_c66x_impl::work(int noutput_items,
        gr_vector_const_void_star &input_items,
        gr_vector_void_star &output_items)
    {
      const gr_complex *in = (const gr_complex *) input_items[0];
      gr_complex *out = (gr_complex *) output_items[0];
      int count, index, iterations;
      int items = noutput_items;
      int iteration_output_items[16];

      unsigned int input_data_size = input_signature()->sizeof_stream_item (0);
      unsigned int output_data_size = output_signature()->sizeof_stream_item (0);

      iterations = (noutput_items / FFTCHS) + 1;
      for (int i = 0; i < iterations; i++) {
        if (items >= FFTCHS) {
          iteration_output_items[i] = FFTCHS;
        }
        else {
          iteration_output_items[i] = items;
        }
        items -= FFTCHS;
      }
      for (int i = 0; i < iterations; i++) {
        count = 0;
        index = 0;
        while (count++ < iteration_output_items[i]) {
          // copy input into optimally aligned buffer
          if (d_window.size()) {
            if (!d_forward && d_shift) {
              unsigned int offset = (!d_forward && d_shift) ? (d_fft_size / 2) : 0;
              int fft_m_offset = d_fft_size - offset;
              volk_32fc_32f_multiply_32fc(&x[fft_m_offset + index], &in[0], &d_window[0], offset);
              volk_32fc_32f_multiply_32fc(&x[index], &in[offset], &d_window[offset], d_fft_size - offset);
            }
            else {
              volk_32fc_32f_multiply_32fc(&x[index], in, &d_window[0], d_fft_size);
            }
          }
          else {
            if (!d_forward && d_shift) {  // apply an ifft shift on the data
              unsigned int len = (unsigned int)(floor(d_fft_size / 2.0)); // half length of complex array
              memcpy(&x[index], &in[len], sizeof(gr_complex) * (d_fft_size - len));
              memcpy(&x[(d_fft_size - len) + index], &in[0], sizeof(gr_complex) * len);
            }
            else {
              memcpy(&x[index], in, input_data_size);
            }
          }
          index += d_fft_size;
          in += d_fft_size;
        }

        // compute the fft
        fft.setArg(0, d_fft_size);
        fft.setArg(1, bufX);
        fft.setArg(2, bufW);
        fft.setArg(3, bufY);
        fft.setArg(4, FFTRADIX);
        fft.setArg(5, 0);
        fft.setArg(6, d_fft_size);
        fft.setArg(7, iteration_output_items[i]);

        cl::Event e1;
        Q.enqueueNDRangeKernel(fft, cl::NullRange, cl::NDRange(NUMCOMPUNITS),
                               cl::NDRange(1), NULL, &e1);
        e1.wait();

        count = 0;
        index = 0;
        while (count++ < iteration_output_items[i]) {
          // copy result to our output
          if (d_forward && d_shift) {  // apply a fft shift on the data
            unsigned int len = (unsigned int)(ceil(d_fft_size / 2.0));
            memcpy(&out[0], &y[len + index], sizeof(gr_complex) * (d_fft_size - len));
            memcpy(&out[d_fft_size - len], &y[index], sizeof(gr_complex) * len);
          }
          else {
            if (!d_forward) {
              // match fftw behavior
              volk_32fc_s32fc_multiply_32fc(out, &y[index], (float)(d_fft_size), d_fft_size);
            }
            else {
              memcpy(out, &y[index], output_data_size);
            }
          }
          index += d_fft_size;
          out += d_fft_size;
        }
      }

      return noutput_items;
    }

  } /* namespace c66xfft */
} /* namespace gr */

