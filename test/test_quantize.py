from pathlib import Path
import sys

import torchvision.models

from tqdm import tqdm
 
# setting path
sys.path.append('../edge_profile')

from model_manager import ModelManager, QuantizedModelManager

paths = ModelManager.getModelPaths()

for path in paths:
    a = QuantizedModelManager(path,  save_model=False)