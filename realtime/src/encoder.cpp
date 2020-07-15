#include "../libwavae/src/deepAudioEngine.h"
#include "cstring"
#include "dlfcn.h"
#include "m_pd.h"
#include "pthread.h"
#include "sched.h"
#include "thread"
#include <iostream>
#include <stdio.h>

#define DAE DeepAudioEngine

static t_class *encoder_tilde_class;

typedef struct _encoder_tilde {
  t_object x_obj;
  t_sample f;

  // OBJECT ATTRIBUTES
  int latent_number, buffer_size, activated;
  float *in_buffer, *out_buffer;
  std::thread *worker;
  DAE *model;

  // DSP RELATED MEMORY MAPS
  float *dsp_in_vec, **dsp_out_vec;
  int dsp_vec_size;

} t_encoder_tilde;

void perform(t_encoder_tilde *x) {
  // SET THREAD TO REALTIME PRIORITY
  pthread_t this_thread = pthread_self();
  struct sched_param params;
  params.sched_priority = sched_get_priority_max(SCHED_FIFO);
  int ret = pthread_setschedparam(this_thread, SCHED_FIFO, &params);

  // COMPUTATION
  x->model->perform(x->in_buffer, x->out_buffer, x->buffer_size);
}

t_int *encoder_tilde_perform(t_int *w) {
  t_encoder_tilde *x = (t_encoder_tilde *)w[1];

  if (x->dsp_vec_size != x->buffer_size) {
    char error[80];
    sprintf(error, "encoder: expecting buffer %d, got %d", x->buffer_size,
            x->dsp_vec_size);
    post(error);
    for (int d(0); d < x->latent_number; d++) {
      for (int i(0); i < x->dsp_vec_size; i++) {
        x->dsp_out_vec[d][i] = 0;
      }
    }
  } else if (x->activated == 0) {
    for (int d(0); d < x->latent_number; d++) {
      for (int i(0); i < x->dsp_vec_size; i++) {
        x->dsp_out_vec[d][i] = 0;
      }
    }
  } else {
    // WAIT FOR PREVIOUS PROCESS TO END
    if (x->worker) {
      x->worker->join();
    }

    // COPY INPUT BUFFER TO OBJECT
    memcpy(x->in_buffer, x->dsp_in_vec, x->dsp_vec_size * sizeof(float));

    // COPY PREVIOUS OUTPUT BUFFER TO PD
    for (int d(0); d < x->latent_number; d++) {
      memcpy(x->dsp_out_vec[d], x->out_buffer + (d * x->dsp_vec_size),
             x->dsp_vec_size * sizeof(float));
    }

    // START NEXT COMPUTATION
    x->worker = new std::thread(perform, x);
  }
  return w + 2;
}

void encoder_tilde_dsp(t_encoder_tilde *x, t_signal **sp) {
  x->dsp_in_vec = sp[0]->s_vec;
  x->dsp_vec_size = sp[0]->s_n;
  for (int i(0); i < x->latent_number; i++) {
    x->dsp_out_vec[i] = sp[i + 1]->s_vec;
  }
  dsp_add(encoder_tilde_perform, 1, x);
}

void encoder_tilde_free(t_encoder_tilde *x) {
  if (x->worker) {
    x->worker->join();
  }
  delete x->in_buffer;
  delete x->out_buffer;
  delete x->dsp_out_vec;
  delete x->model;
}

void *encoder_tilde_new(t_floatarg latent_number, t_floatarg buffer_size) {
  t_encoder_tilde *x = (t_encoder_tilde *)pd_new(encoder_tilde_class);

  x->latent_number = int(latent_number) == 0 ? 16 : int(latent_number);
  x->buffer_size = int(buffer_size) == 0 ? 512 : int(buffer_size);
  x->activated = 1;

  for (int i(0); i < x->latent_number; i++) {
    outlet_new(&x->x_obj, &s_signal);
  }

  x->in_buffer = new float[x->buffer_size];
  x->out_buffer = new float[x->latent_number * x->buffer_size];

  x->worker = NULL;

  void *hndl = dlopen("/usr/lib/libwavae.so", RTLD_LAZY);
  if (!hndl) {
    hndl = dlopen("./libwavae/libwavae.so", RTLD_LAZY);
    post("Using local version of libwavae");
  }

  x->model = reinterpret_cast<DAE *(*)()>(dlsym(hndl, "get_encoder"))();
  x->model->set_latent_number(x->latent_number);

  x->dsp_out_vec = new float *[x->latent_number];

  return (void *)x;
}

void encoder_tilde_load(t_encoder_tilde *x, t_symbol *sym) {
  int statut = x->model->load(sym->s_name);

  if (statut == 0) {
    post("encoder loaded");
  } else {
    post("encoder failed loading model");
  }
}

void encoder_tilde_activate(t_encoder_tilde *x, t_floatarg arg) {
  x->activated = int(arg);
}

extern "C" {
void encoder_tilde_setup(void) {
  encoder_tilde_class =
      class_new(gensym("encoder~"), (t_newmethod)encoder_tilde_new, 0,
                sizeof(t_encoder_tilde), 0, A_DEFFLOAT, A_DEFFLOAT, 0);

  class_addmethod(encoder_tilde_class, (t_method)encoder_tilde_dsp,
                  gensym("dsp"), A_CANT, 0);
  class_addmethod(encoder_tilde_class, (t_method)encoder_tilde_load,
                  gensym("load"), A_SYMBOL, A_NULL);
  class_addmethod(encoder_tilde_class, (t_method)encoder_tilde_activate,
                  gensym("activate"), A_DEFFLOAT, A_NULL);

  CLASS_MAINSIGNALIN(encoder_tilde_class, t_encoder_tilde, f);
}
}
