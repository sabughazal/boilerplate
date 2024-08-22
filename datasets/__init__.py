from .acme_dataset import AcmeDataset
DATASETS = {}

def get_AcmeDataset(data_root, split, cfg=None):
    return AcmeDataset(
        data_root=data_root,
        split=split
    )
DATASETS["Acme"] = get_AcmeDataset
