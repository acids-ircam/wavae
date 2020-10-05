import torch
torch.set_grad_enabled(False)


import librosa as li
import numpy as np
import soundfile as sf

model = torch.jit.load("runs/test_idx/test_idx_16kHz.ts")

x = li.load("voice.wav", 16000)[0]

N = 2**int(np.ceil(np.log2(len(x))))
x = np.pad(x, (0, N - len(x)))

x = torch.from_numpy(x).float().reshape(1, -1)

x = x.expand(5, x.shape[-1])
t = torch.arange(5)
y = model.decode(x, t).reshape(-1).numpy()

sf.write("rec.wav", y, 16000)
