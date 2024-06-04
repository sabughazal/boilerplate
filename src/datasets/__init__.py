from .gravity_anomaly import GravityAnomalyDataset

def get_GravityAnomalyDataset(data_root, split, cfg=None):
    return GravityAnomalyDataset(
        data_root=data_root,
        split=split
    )

DATASETS = {
    "GravityAnomaly": get_GravityAnomalyDataset
}
