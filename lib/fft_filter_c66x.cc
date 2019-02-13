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

#include <c66xfft/fft_filter_c66x.h>
#include <volk/volk.h>
#include "fft_vcc_c66x_impl.h"

extern char kernel_dsp_bin[15736];

#define FFTCHS (64)

namespace gr {
  namespace c66xfft {
    namespace kernel {

      fft_filter_ccc_c66x::fft_filter_ccc_c66x(int decimation, const std::vector<gr_complex> &taps)
        : d_fftsize(-1), d_decimation(decimation),
        d_xformed_taps(NULL),
        w(NULL),
        x(NULL),
        y(NULL),
        z(NULL)
      {
        set_taps(taps);
      }

      fft_filter_ccc_c66x::~fft_filter_ccc_c66x()
      {
        if (d_xformed_taps != NULL) {
          volk_free(d_xformed_taps);
        }
        if (z != NULL) {
          __free_ddr(z);
        }
        if (w != NULL) {
          __free_ddr(w);
        }
        if (y != NULL) {
          __free_ddr(y);
        }
        if (x != NULL) {
          __free_ddr(x);
        }
      }

      /*
       * determines d_ntaps, d_nsamples, d_fftsize, d_xformed_taps
       */
      int
      fft_filter_ccc_c66x::set_taps(const std::vector<gr_complex> &taps)
      {
        int i = 0;
        d_taps = taps;
        compute_sizes(taps.size());

        d_tail.resize(tailsize());
        for (i = 0; i < tailsize(); i++) {
          d_tail[i] = 0;
        }

        gr_complex *in = &x[0];
        gr_complex *out = &y[0];

        float scale = 1.0 / d_fftsize;

        // Compute forward xform of taps.
        // Copy taps into first ntaps slots, then pad with zeros
        for (i = 0; i < d_ntaps; i++) {
          in[i] = taps[i] * scale;
        }

        for (; i < d_fftsize; i++) {
          in[i] = 0;
        }

        // do the xform
        fwdfft.setArg(0, d_fftsize);
        fwdfft.setArg(1, bufX);
        fwdfft.setArg(2, bufW);
        fwdfft.setArg(3, bufY);
        fwdfft.setArg(4, FFTRADIX);
        fwdfft.setArg(5, 0);
        fwdfft.setArg(6, d_fftsize);
        fwdfft.setArg(7, 1);

        cl::Event e1;
        Q.enqueueNDRangeKernel(fwdfft, cl::NullRange, cl::NDRange(NUMCOMPUNITS),
                               cl::NDRange(1), NULL, &e1);
        e1.wait();

        // now copy output to d_xformed_taps
        for (i = 0; i < d_fftsize; i++) {
          d_xformed_taps[i] = out[i];
        }

        return d_nsamples;
      }

      // determine and set d_ntaps, d_nsamples, d_fftsize
      void
      fft_filter_ccc_c66x::compute_sizes(int ntaps)
      {
        int old_fftsize = d_fftsize;
        d_ntaps = ntaps;
        d_fftsize = (int) (2 * pow(2.0, ceil(log(double(ntaps)) / log(2.0))));
        d_nsamples = d_fftsize - d_ntaps + 1;

        if (0) {
          std::cerr << "fft_filter_ccc_c66x: ntaps = " << d_ntaps
                    << " fftsize = " << d_fftsize
                    << " nsamples = " << d_nsamples << std::endl;
        }

        if(d_fftsize != old_fftsize) {
          if (z != NULL) {
            __free_ddr(z);
          }
          if (w != NULL) {
            __free_ddr(w);
          }
          if (y != NULL) {
            __free_ddr(y);
          }
          if (x != NULL) {
            __free_ddr(x);
          }
          if (d_fftsize & (d_fftsize - 1)) {
            throw std::runtime_error("fft_vcc_c66: FFT Size not a power of 2\n");
          }
          if (d_fftsize & 0x55555555) {
            FFTRADIX = 4;
          }
          else {
            FFTRADIX = 2;
          }
          int channel_size = d_fftsize * sizeof(gr_complex);
          // __malloc_ddr() returns 128 bytes aligned memory
          x = (gr_complex *) __malloc_ddr(FFTCHS * channel_size);
          y = (gr_complex *) __malloc_ddr(FFTCHS * channel_size);
          // same twiddle factor for all channels of same size
          w = (float *) __malloc_ddr(channel_size);
          z = (float *) __malloc_ddr(channel_size);
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
          bufZ = cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
                            channel_size, z);

          binary = cl::Program::Binaries(1, std::make_pair(kernel_dsp_bin,
                                         sizeof(kernel_dsp_bin)));
          program = cl::Program(context, devices, binary);
          program.build(devices);
          kernel = cl::Kernel(program, "null");
          null = kernel.bind(Q, cl::NDRange(1), cl::NDRange(1));
          null().wait();
          tw_gen_forward(w, d_fftsize);  // Generate twiddle factors
          fwdfft = cl::Kernel(program, "ocl_DSPF_sp_fftSPxSP");
          tw_gen_reverse(z, d_fftsize);  // Generate twiddle factors
          invfft = cl::Kernel(program, "ocl_DSPF_sp_ifftSPxSP");
        }
        if (d_fftsize != old_fftsize) {
          if (d_xformed_taps != NULL) {
            volk_free(d_xformed_taps);
          }
	  d_xformed_taps = (gr_complex*)volk_malloc(sizeof(gr_complex)*d_fftsize,
                                                    volk_get_alignment());
        }
      }

      std::vector<gr_complex>
      fft_filter_ccc_c66x::taps() const
      {
        return d_taps;
      }

      unsigned int
      fft_filter_ccc_c66x::ntaps() const
      {
        return d_ntaps;
      }

      void
      fft_filter_ccc_c66x::tw_gen_forward(float *w, int n)
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
      fft_filter_ccc_c66x::tw_gen_reverse(float *w, int n)
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
      fft_filter_ccc_c66x::filter(int nitems, const gr_complex *input, gr_complex *output)
      {
        int dec_ctr = 0;
        int j = 0;
        int ninput_items = nitems * d_decimation;
        int indexin, indexout, iterations;
        int items = (ninput_items / d_nsamples);
        int iteration_input_items[16];

        iterations = ((ninput_items / d_nsamples) / FFTCHS) + 1;
        for (int i = 0; i < iterations; i++) {
          if (items >= FFTCHS) {
            iteration_input_items[i] = FFTCHS;
          }
          else {
            iteration_input_items[i] = items;
          }
          items -= FFTCHS;
        }
        for (int n = 0; n < iterations; n++) {
          indexin = 0;
          indexout = 0;
          for (int i = 0; i < iteration_input_items[n]; i++) {
            memcpy(&x[indexout], &input[indexin], d_nsamples * sizeof(gr_complex));

            for (j = d_nsamples; j < d_fftsize; j++) {
              x[j + indexout] = 0;
            }
            indexin += d_nsamples;
            indexout += d_fftsize;
          }

          // compute fwd xform
          fwdfft.setArg(0, d_fftsize);
          fwdfft.setArg(1, bufX);
          fwdfft.setArg(2, bufW);
          fwdfft.setArg(3, bufY);
          fwdfft.setArg(4, FFTRADIX);
          fwdfft.setArg(5, 0);
          fwdfft.setArg(6, d_fftsize);
          fwdfft.setArg(7, iteration_input_items[n]);

          cl::Event e2;
          Q.enqueueNDRangeKernel(fwdfft, cl::NullRange, cl::NDRange(NUMCOMPUNITS),
                                 cl::NDRange(1), NULL, &e2);
          e2.wait();

          gr_complex *b = d_xformed_taps;

          indexout = 0;
          for (int i = 0; i < iteration_input_items[n]; i++) {
            volk_32fc_x2_multiply_32fc_a(&x[indexout], &y[indexout], b, d_fftsize);
            indexout += d_fftsize;
          }

          // compute inv xform
          invfft.setArg(0, d_fftsize);
          invfft.setArg(1, bufX);
          invfft.setArg(2, bufZ);
          invfft.setArg(3, bufY);
          invfft.setArg(4, FFTRADIX);
          invfft.setArg(5, 0);
          invfft.setArg(6, d_fftsize);
          invfft.setArg(7, iteration_input_items[n]);

          cl::Event e3;
          Q.enqueueNDRangeKernel(invfft, cl::NullRange, cl::NDRange(NUMCOMPUNITS),
                                 cl::NDRange(1), NULL, &e3);
          e3.wait();

          // match fftw behavior
          volk_32fc_s32fc_multiply_32fc(&x[0], &y[0], (float)(d_fftsize), d_fftsize * iteration_input_items[n]);

          indexout = 0;
          for (int i = 0; i < iteration_input_items[n]; i++) {
            // add in the overlapping tail
            for (j = 0; j < tailsize(); j++) {
              x[j + indexout] += d_tail[j];
            }

            // copy nsamples to output
            j = dec_ctr;
            while (j < d_nsamples) {
              *output++ = x[j + indexout];
              j += d_decimation;
            }
            dec_ctr = (j - d_nsamples);

            // stash the tail
            if (d_tail.size()) {
              memcpy(&d_tail[0], &x[indexout + d_nsamples], tailsize() * sizeof(gr_complex));
            }
            indexout += d_fftsize;
          }
        }

        return nitems;
      }

    } /* namespace kernel */
  } /* namespace c66xfft */
} /* namespace gr */
