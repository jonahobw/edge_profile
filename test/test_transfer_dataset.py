import sys

# setting path
sys.path.append('../edge_profile')

from model_manager import VictimModelManager

arch = "alexnet"
path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]

manager = VictimModelManager.load(path)

dataset = "cifar100"
transfer_size = 600
sample_average = 50
random = False

manager.generateKnockoffTransferSet(dataset_name=dataset, transfer_size=transfer_size, sample_avg=sample_average, random_policy=random)
# file, dataset = manager.loadKnockoffTransferSet(dataset_name=dataset, transfer_size=transfer_size, sample_avg=sample_average, random_policy=random)

# print(file)
# print(dataset.config)