#include "../libwavae/src/deepAudioEngine.h"
#include "cstring"
#include "dlfcn.h"
#include "m_pd.h"
#include "pthread.h"
#include "sched.h"
#include "thread"
#include <iostream>

#define DAE DeepAudioEngine

static t_class *decoder_tilde_class;

typedef struct _decoder_tilde {
  t_object x_obj;
  t_sample f;

  // OBJECT ATTRIBUTES
  int loaded, latent_number, buffer_size, activated;
  float *in_buffer, *out_buffer, fadein;
  std::thread *worker;
  DAE *model;

  // DSP RELATED MEMORY MAPS
  float **dsp_in_vec, *dsp_out_vec;
  int dsp_vec_size;

} t_decoder_tilde;

void perform(t_decoder_tilde *x) {
  // SET THREAD TO REALTIME PRIORITY
  pthread_t this_thread = pthread_self();
  struct sched_param params;
  params.sched_priority = sched_get_priority_max(SCHED_FIFO);
  int ret = pthread_setschedparam(this_thread, SCHED_FIFO, &params);

  // COMPUTATION
  x->model->perform(x->in_buffer, x->out_buffer, x->buffer_size);
}

t_int *decoder_tilde_perform(t_int *w) {
  t_decoder_tilde *x = (t_decoder_tilde *)w[1];
  if (x->dsp_vec_size != x->buffer_size) {
    char error[80];
    sprintf(error, "decoder: expecting buffer %d, got %d", x->buffer_size,
            x->dsp_vec_size);
    post(error);
    for (int i(0); i < x->dsp_vec_size; i++) {
      x->dsp_out_vec[i] = 0;
    }
  } else if (x->activated == 0) {
    for (int i(0); i < x->dsp_vec_size; i++) {
      x->dsp_out_vec[i] = 0;
    }
  } else {
    // WAIT FOR PREVIOUS PROCESS TO END
    if (x->worker) {
      x->worker->join();
    }

    // COPY INPUT BUFFER TO OBJECT
    for (int d(0); d < x->latent_number; d++) {
      memcpy(x->in_buffer + (d * x->buffer_size), x->dsp_in_vec[d],
             x->buffer_size * sizeof(float));
    }

    // COPY PREVIOUS OUTPUT BUFFER TO PD
    memcpy(x->dsp_out_vec, x->out_buffer, x->buffer_size * sizeof(float));

    // FADE IN
    if (x->fadein < .99) {
      for (int i(0); i < x->buffer_size; i++) {
        x->dsp_out_vec[i] *= x->fadein;
        x->fadein = x->loaded ? x->fadein * .99999 + 0.00001 : x->fadein;
      }
    }

    // START NEXT COMPUTATION
    x->worker = new std::thread(perform, x);
  }
  return w + 2;
}

void decoder_tilde_dsp(t_decoder_tilde *x, t_signal **sp) {
  x->dsp_vec_size = sp[0]->s_n;
  for (int i(0); i < x->latent_number; i++) {
    x->dsp_in_vec[i] = sp[i]->s_vec;
  }
  x->dsp_out_vec = sp[x->latent_number]->s_vec;
  dsp_add(decoder_tilde_perform, 1, x);
}

void decoder_tilde_free(t_decoder_tilde *x) {
  if (x->worker) {
    x->worker->join();
  }
  delete x->in_buffer;
  delete x->out_buffer;
  delete x->dsp_in_vec;
  delete x->model;
}

void *decoder_tilde_new(t_floatarg latent_number, t_floatarg buffer_size) {
  t_decoder_tilde *x = (t_decoder_tilde *)pd_new(decoder_tilde_class);

  x->latent_number = int(latent_number) == 0 ? 16 : int(latent_number);
  x->buffer_size = int(buffer_size) == 0 ? 512 : int(buffer_size);
  x->activated = 1;

  outlet_new(&x->x_obj, &s_signal);
  for (int i(1); i < x->latent_number; i++) {
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
  }

  x->in_buffer = new float[x->latent_number * x->buffer_size];
  x->out_buffer = new float[x->buffer_size];

  x->worker = NULL;

  x->loaded = 0;
  x->fadein = 0;

  void *hndl = dlopen("/usr/lib/libwavae.so", RTLD_LAZY);
  if (!hndl) {
    hndl = dlopen("./libwavae/libwavae.so", RTLD_LAZY);
    post("Using local version of libwavae");
  }

  x->model = reinterpret_cast<DAE *(*)()>(dlsym(hndl, "get_decoder"))();
  x->model->set_latent_number(x->latent_number);

  x->dsp_in_vec = new float *[x->latent_number];

  return (void *)x;
}

void decoder_tilde_load(t_decoder_tilde *x, t_symbol *sym) {
  x->loaded = 0;
  x->fadein = 0;

  int statut = x->model->load(sym->s_name);

  if (statut == 0) {
    x->loaded = 1;
    post("decoder loaded");
  } else {
    post("decoder failed loading model");
  }
}

void decoder_tilde_activate(t_decoder_tilde *x, t_floatarg arg) {
  x->activated = int(arg);
}

extern "C" {
void decoder_tilde_setup(void) {
  decoder_tilde_class =
      class_new(gensym("decoder~"), (t_newmethod)decoder_tilde_new, 0,
                sizeof(t_decoder_tilde), 0, A_DEFFLOAT, A_DEFFLOAT, 0);

  class_addmethod(decoder_tilde_class, (t_method)decoder_tilde_dsp,
                  gensym("dsp"), A_CANT, 0);
  class_addmethod(decoder_tilde_class, (t_method)decoder_tilde_load,
                  gensym("load"), A_SYMBOL, A_NULL);
  class_addmethod(decoder_tilde_class, (t_method)decoder_tilde_activate,
                  gensym("activate"), A_DEFFLOAT, A_NULL);

  CLASS_MAINSIGNALIN(decoder_tilde_class, t_decoder_tilde, f);
}
}
