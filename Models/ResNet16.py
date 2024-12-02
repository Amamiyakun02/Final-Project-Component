import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class ResNet16:
    def __init__(self):
        model = models.resnet16(pretrained=True)