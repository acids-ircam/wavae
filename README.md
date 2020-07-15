# WaVAE

Cause I love naming models <3

Despite its name, its not a waveform based VAE, but a melspec one with a melGAN decoder. There is also an adversarial regularization on the latent space in order to extract the loudness from it. That's actually pretty cool ! You can even use it on your dataset, and train the whole thing in maybe 2-4 days on a single GPU.

This model has realtime generation (on CPU, even if it can burn it lol, so you'd better stick to GPU) and a highly-compressed and expressive latent representation.

## PureData usage demo

[![Celine to Scream](https://img.youtube.com/vi/Q3Ejm_ll6KU/0.jpg)](https://www.youtube.com/watch?v=Q3Ejm_ll6KU)


## Usage

Train the spectral model
```bash
python train.py -c vanilla --wav-loc YOUR_DATA_FOLDER --name ENTER_A_COOL_NAME
```

Remember to delete the `preprocessed` folder between each training, as the models don't use the same preprocessing pipeline. (You can also outsmart us all and use the `--lmdb-loc` flag with a different path for each model)

Train the waveform model
```bash
python train.py -c melgan --wav-loc YOUR_DATA_FOLDER --name ENTER_THE_SAME_COOL_NAME
```

The training scripts logs into the `runs` folder, you can visualize it using `tensorboard`.


Onced both models are trained, trace them using
```bash
python make_wrapper.py --name AGAIN_THE_SAME_COOL_NAME
```

It will produce a traced script in `runs/COOL_NAME/COOLNAME_LOTSOFWEIRDNUMBERS.ts`. It can be deployed, used in a libtorch C++ environement, inside a Max/MSP playground that won't be named here, without having to use the source code. AND if you want to use the swaggy realtime abilities of this model, just pass the `--use-cached-padding true --buffer-size 2048`.

## Compiling


To compile the pd externals, you can use CMAKE
```bash
cmake -DCMAKE_PREFIX_PATH=/path.to.libtorch -DCMAKE_BUILD_TYPE=[Release / Debug] -DCUDNN_LIBRARY_PATH=path.to.libcudnn.so -DCUDNN_INCLUDE_PATH=path.to.cudnn.include -G [Ninja / Xcode / Makefile]  ../
```

Or even better, use the precompiled binaries available in the **Release** section of this project.
Just remember to download the CUDA 10.1 cxx11 ABI version of libtorch and unzip it in `/usr/lib/`

(only tested on ubuntu 18.04 - 19.10 - 20.04)
