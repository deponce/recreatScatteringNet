'''
This class defines all of the models that we will be using for the
experiments that we have to run
'''
import torch
import torchvision.models as models
from .scatter_resnet import scatternet_cnn

model_dict = {
    "alexnet": models.alexnet(),
    "scatternet": None,
    "scatter_resnet": scatternet_cnn(),
    "resnet18": models.resnet18()
}
