import sys
 
# setting path
sys.path.append('../edge_profile')

import datasets

a = datasets.Dataset("cifar10", data_subset_percent=0.1)
print(len(a.train_data))
print(len(a.val_data))
print(len(a.train_acc_data))

assert len(a.train_acc_data) == len(a.train_data)
assert a.train_acc_data.indices == a.train_data.indices

b = datasets.Dataset("cifar10", data_subset_percent=0.1, idx=1)
print(len(b.train_data))
print(len(b.val_data))
print(len(b.train_acc_data))

assert len(b.train_acc_data) == len(b.train_data)
assert b.train_acc_data.indices == b.train_data.indices

c = datasets.Dataset("cifar10")
print(len(c.train_data))
print(len(c.val_data))
print(len(c.train_acc_data))

assert len(c.train_data) == len(a.train_data) + len(b.train_data)
assert len(c.val_data) == len(a.val_data) + len(b.val_data)
assert len(c.train_acc_data) == len(a.train_acc_data) + len(b.train_acc_data)

assert not set(b.train_data.indices).intersection(set(a.train_data.indices))
assert not set(b.val_data.indices).intersection(set(a.val_data.indices))


# check for repetition
d = datasets.Dataset("cifar10", data_subset_percent=0.1)
assert d.train_data.indices == a.train_data.indices
assert d.val_data.indices == a.val_data.indices

e = datasets.Dataset("cifar10", data_subset_percent=0.1, idx=1)
assert e.train_data.indices == b.train_data.indices
assert e.val_data.indices == b.val_data.indices


print("All checks valid")