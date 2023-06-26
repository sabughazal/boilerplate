from models.cnn import CNN

def get_CNN(cfg):
    return CNN(cfg.MODEL.INPUT_SIZE, cfg.MODEL.NUM_CLASSES)

MODELS = {
    "CNN": get_CNN
}