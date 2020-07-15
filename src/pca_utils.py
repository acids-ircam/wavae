import torch
from . import Loader, config
from tqdm import tqdm


def compute_pca(model, batch_size):
    print(config)
    loader = Loader(5)
    dataloader = torch.utils.data.DataLoader(loader,
                                             batch_size=batch_size,
                                             drop_last=False,
                                             shuffle=True)

    z = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for elm in tqdm(dataloader, desc="parsing dataset..."):
        z_ = model.encode(elm[0].squeeze().to(device))  # SHAPE B x Z x T
        z_ = z_.permute(0, 2, 1).reshape(-1, z_.shape[1]).cpu()  # SHAPE BT x Z
        z.append(z_)

    z = torch.cat(z, 0)
    z = z[torch.randperm(z.shape[0])][:10000].permute(1, 0)
    z = z[:, torch.max(z, 0)[0] < 10]

    mean = torch.mean(z, -1, keepdim=True)
    std = 3 * torch.std(z)  # 99.7% of the range (normal law)
    U = torch.svd(z - mean, some=False)[0]
    # U = torch.svd(z, some=False)[0]

    # torch.save(z, "z.pth")
    # torch.save(mean, "mean.pth")
    # torch.save(std, "std.pth")
    # torch.save(U, "U.pth")

    return mean.reshape(1, 1, -1), std, U
