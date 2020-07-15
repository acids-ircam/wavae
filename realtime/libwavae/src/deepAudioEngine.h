#pragma once
#include <string>
#include <vector>

#define DIM_REDUCTION_FACTOR 512

class DeepAudioEngine {
public:
  virtual void perform(float *in_buffer, float *out_buffer,
                       int dsp_vec_size) = 0;
  virtual int load(std::string name) = 0;
  virtual void set_latent_number(int n) = 0;
};