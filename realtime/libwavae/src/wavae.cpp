#include "wavae.h"
#include "deepAudioEngine.h"
#include <iostream>
#include <stdlib.h>

#define DEVICE torch::kCUDA
#define CPU torch::kCPU

// ENCODER /////////////////////////////////////////////////////////

wavae::Encoder::Encoder() {
  model_loaded = 0;
  at::init_num_threads();
}

void wavae::Encoder::set_latent_number(int n) { latent_number = n; }

void wavae::Encoder::perform(float *in_buffer, float *out_buffer,
                             int dsp_vec_size) {
  torch::NoGradGuard no_grad;

  if (model_loaded) {

    auto tensor = torch::from_blob(in_buffer, {1, dsp_vec_size});
    tensor = tensor.to(DEVICE);

    std::vector<torch::jit::IValue> input;
    input.push_back(tensor);

    auto out_tensor = model.get_method("encode")(std::move(input)).toTensor();

    out_tensor = out_tensor.repeat_interleave(DIM_REDUCTION_FACTOR);
    out_tensor = out_tensor.to(CPU);

    auto out = out_tensor.contiguous().data_ptr<float>();

    for (int i(0); i < latent_number * dsp_vec_size; i++) {
      out_buffer[i] = out[i];
    }

  } else {

    for (int i(0); i < latent_number * dsp_vec_size; i++) {
      out_buffer[i] = 0;
    }
  }
}

int wavae::Encoder::load(std::string name) {
  try {
    model = torch::jit::load(name);
    model.to(DEVICE);
    model_loaded = 1;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

// DECODER /////////////////////////////////////////////////////////

wavae::Decoder::Decoder() {
  model_loaded = 0;
  at::init_num_threads();
}

void wavae::Decoder::set_latent_number(int n) { latent_number = n; }

void wavae::Decoder::perform(float *in_buffer, float *out_buffer,
                             int dsp_vec_size) {

  torch::NoGradGuard no_grad;

  if (model_loaded) {

    auto tensor = torch::from_blob(in_buffer, {1, latent_number, dsp_vec_size});
    tensor =
        tensor.reshape({1, latent_number, -1, DIM_REDUCTION_FACTOR}).mean(-1);
    tensor = tensor.to(DEVICE);

    std::vector<torch::jit::IValue> input;
    input.push_back(tensor);

    auto out_tensor = model.get_method("decode")(std::move(input))
                          .toTensor()
                          .reshape({-1})
                          .contiguous();

    out_tensor = out_tensor.to(CPU);

    auto out = out_tensor.data_ptr<float>();

    for (int i(0); i < dsp_vec_size; i++) {
      out_buffer[i] = out[i];
    }
  } else {
    for (int i(0); i < dsp_vec_size; i++) {
      out_buffer[i] = 0;
    }
  }
}

int wavae::Decoder::load(std::string name) {
  try {
    model = torch::jit::load(name);
    model.to(DEVICE);
    model_loaded = 1;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

extern "C" {
DeepAudioEngine *get_encoder() { return new wavae::Encoder; }
DeepAudioEngine *get_decoder() { return new wavae::Decoder; }
}