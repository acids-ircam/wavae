# Training melgan

The loss definition is quite something, so here is creepy copy-pasta of the original training procedure.

```python
#######################
# Train Discriminator #
#######################
D_fake_det = netD(x_pred_t.cuda().detach())
D_real = netD(x_t.cuda())

loss_D = 0
for scale in D_fake_det:
    loss_D += F.relu(1 + scale[-1]).mean()

for scale in D_real:
    loss_D += F.relu(1 - scale[-1]).mean()

netD.zero_grad()
loss_D.backward()
optD.step()

###################
# Train Generator #
###################
D_fake = netD(x_pred_t.cuda())

loss_G = 0
for scale in D_fake:
    loss_G += -scale[-1].mean()

loss_feat = 0
feat_weights = 4.0 / (args.n_layers_D + 1)
D_weights = 1.0 / args.num_D
wt = D_weights * feat_weights
for i in range(args.num_D):
    for j in range(len(D_fake[i]) - 1):
        loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

netG.zero_grad()
(loss_G + args.lambda_feat * loss_feat).backward()
optG.step()
```

