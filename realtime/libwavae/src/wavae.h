#pragma once
#include "deepAudioEngine.h"
#include <torch/script.h>
#include <torch/torch.h>

namespace wavae {

class Encoder : public DeepAudioEngine {
public:
  Encoder();
  void perform(float *in_buffer, float *out_buffer, int dsp_vec_size) override;
  int load(std::string name) override;
  void set_latent_number(int n) override;

protected:
  int model_loaded;
  int latent_number;
  torch::jit::script::Module model;
};

class Decoder : public DeepAudioEngine {
public:
  Decoder();
  void perform(float *in_buffer, float *out_buffer, int dsp_vec_size) override;
  int load(std::string name) override;
  void set_latent_number(int n) override;

protected:
  int model_loaded;
  int latent_number;
  torch::jit::script::Module model;
};

} // namespace wavae
