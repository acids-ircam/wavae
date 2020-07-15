import torch
import lmdb
import pickle


class SimpleLMDBDataset(torch.utils.data.Dataset):
    """
    Wraps a LDMB database as a torch compatible Dataset
    """
    def __init__(self, out_database_location, map_size=1e9):
        super().__init__()
        self.env = lmdb.open(out_database_location,
                             map_size=map_size,
                             lock=False)
        with self.env.begin(write=False) as txn:
            lmdblength = txn.get("length".encode("utf-8"))
        self.len = int(lmdblength) if lmdblength is not None else 0

    def __len__(self):
        return self.len

    def __setitem__(self, idx, value):
        with self.env.begin(write=True) as txn:
            txn.put(f"{idx:08d}".encode("utf-8"), pickle.dumps(value))
            if idx > self.len - 1:
                self.len = idx + 1
                txn.put("length".encode("utf-8"),
                        f"{self.len:08d}".encode("utf-8"))

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            value = pickle.loads(txn.get(f"{idx:08d}".encode("utf-8")))
        return value