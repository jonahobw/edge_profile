"""
Use this module when you have updated the code in
model_manager.py such that it is not backwards compatible
with previous config formats.
"""

from pathlib import Path
import json
import sys
 
# setting path
sys.path.append('../edge_profile')

from model_manager import VictimModelManager

def addConfig(args: dict):
    """For each model, add <args> to its configuration"""
    for path in VictimModelManager.getModelPaths():
        manager = VictimModelManager.load(path)
        manager.saveConfig(args)

if __name__ == '__main__':
    addConfig({"pretrained": False})
