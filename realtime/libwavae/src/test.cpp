#include "wavae.h"
#include <dlfcn.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

#define LATENT_NUMBER 16
#define BUFFERSIZE 2048

int main(int argc, char const *argv[]) {

  DeepAudioEngine *encoder = new wavae::Encoder;
  int error = encoder->load("trace_model.ts");

  DeepAudioEngine *decoder = new wavae::Decoder;
  error = decoder->load("trace_model.ts");

  float *inbuffer = new float[BUFFERSIZE];
  float *outbuffer = new float[BUFFERSIZE];
  float *zbuffer = new float[LATENT_NUMBER * BUFFERSIZE / DIM_REDUCTION_FACTOR];

  // LOOP TEST
  for (int i(0); i < 100; i++) {
    std::cout << i << std::endl;
    encoder->perform(inbuffer, zbuffer, BUFFERSIZE);
    decoder->perform(zbuffer, outbuffer, BUFFERSIZE);
  }

  return 0;
}
