import torch

model = torch.jit.load("runs/untitled/trace_model.ts")

x = torch.randn(1, 2048)

model.encode(x)