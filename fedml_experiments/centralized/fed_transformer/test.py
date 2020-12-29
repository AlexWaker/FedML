import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.distributed.fed_transformer.utils import count_parameters
from fedml_api.model.cv.transformer.vit.vision_transformer_origin import VisionTransformer, CONFIGS

# CIFAR10: 32*32*3
pretrained_dir = "./../../../fedml_api/model/cv/pretrained/Transformer/vit/ViT-B_16.npz"
img_size = 224
model_type = 'vit-B_16'
# pretrained on ImageNet (224x224), and fine-tuned on (384x384) high resolution.
config = CONFIGS[model_type]
logging.info("Vision Transformer Configuration: " + str(config))
num_classes = 10
model = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes)
model.load_from(np.load(pretrained_dir))
num_params = count_parameters(model)
logging.info("Vision Transformer Model Size = " + str(num_params))
print(model)