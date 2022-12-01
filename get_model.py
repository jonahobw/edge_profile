import torchvision.models as models

alexnet = "alexnet", ['alexnet']
resnet = "resnet", ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d',
          'resnext101_32x8d','wide_resnet50_2', 'wide_resnet101_2']
vgg = "vgg", ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']
squeezenet = "squeezenet", ['squeezenet1_0', 'squeezenet1_1']
densenet = "densenet", ['densenet121', 'densenet169', 'densenet201', 'densenet161']
googlenet = "googlenet", ['googlenet']
mobilenet = "mobilenet", ['mobilenet_v2', "mobilenet_v3_large", "mobilenet_v3_small"]
mnasnet = "mnasnet", ['mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3']
shufflenet = "shufflenet", ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0']

model_families = [alexnet, resnet, vgg, squeezenet, densenet, googlenet, mobilenet, mnasnet, shufflenet]

name_to_family = {}
for family, family_models in model_families:
    for model in family_models:
        name_to_family[model] = family

all_models = []
for i in model_families:
    all_models.extend(i[1])

model_params = {
    "googlenet": {
        "kwargs": {
            "aux_logits": False
        }
    },
    "alexnet": {
        "input_size": 224
    }
}

def get_model(model_arch:str, pretrained=False, kwargs={}):
    model_arch = model_arch.lower()
    if model_arch not in name_to_family:
        raise ValueError(f"Model {model_arch} not supported")
    if model_arch in model_params:
        kwargs.update(model_params.get(model_arch, {}).get("kwargs", {}))
    return getattr(models, model_arch)(pretrained=pretrained, **kwargs)