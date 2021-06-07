# %%
import torch
torch.set_grad_enabled(False)

model = torch.jit.load("runs/screams/screams_48kHz_16z_4096b.ts")
x = torch.randn(1,4096)

z = model.encode(x)
# %%
y = model.decode(z)
# %%
print(y.shape)
# %%
