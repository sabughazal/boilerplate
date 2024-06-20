from .acme_model import AcmeModel

def get_AcmeModel(cfg):
    return AcmeModel(cfg.MODEL.INPUT_SIZE, cfg.MODEL.NUM_CLASSES)

MODELS = {
    "Acme": get_AcmeModel
}
