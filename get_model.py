import torchvision.models as models
import torch

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

quantized_models = [x for x in list(name_to_family.keys()) if hasattr(models.quantization, x)]

def getModelParams(model_arch: str):
    model_params = {
        "googlenet": {
            "kwargs": {
                "aux_logits": False
            }
        },
        "alexnet": {
            "input_size": 224,
            "lr": 0.01
        },
        "resnext50_32x4d": {
            "lr": 0.01
        },
        "resnext101_32x8d": {
            "lr": 0.01
        },
        "squeezenet1_0": {
            "lr": 0.0001,
            "optim": "adam"
        },
        "squeezenet1_1": {
            "lr": 0.0001,
            "optim": "adam"
        },
        "vgg11": {
            "lr": 0.01
        }, 
        "vgg13": {
            "lr": 0.01
        }, 
        "vgg16": {
            "lr": 0.01
        }, 
        "vgg19": {
            "lr": 0.01
        },   
        "vgg11_bn": {
            "lr": 0.01
        }, 
        "vgg13_bn": {
            "lr": 0.01
        }, 
        "vgg16_bn": {
            "lr": 0.01
        }, 
        "vgg19_bn": {
            "lr": 0.01
        },
        "mnasnet0_5": {
            "lr": 0.001,
            "input_size": 224
        }, 
        "mnasnet0_75": {
            "lr": 0.001,
            "input_size": 224
        }, 
        "mnasnet1_0": {
            "lr": 0.001,
            "input_size": 224
        }, 
        "mnasnet1_3": {
            "lr": 0.001,
            "input_size": 224
        }, 
    }
    return model_params.get(model_arch, {})

def get_model(model_arch:str, pretrained=False, kwargs={}):
    """If pretrained is true, does not pass kwargs"""
    model_arch = model_arch.lower()
    if model_arch not in name_to_family:
        raise ValueError(f"Model {model_arch} not supported")
    model_params = getModelParams(model_arch)
    if "kwargs" in model_params and not pretrained:
        kwargs.update(model_params["kwargs"])
    if "num_classes" in kwargs and pretrained:
        num_classes = kwargs["num_classes"]
        kwargs.pop("num_classes")
        model = getattr(models, model_arch)(pretrained=pretrained, **kwargs)
        success = fixLastLayer(model, model_arch, num_classes)
        if not success:
            print("Cannot reset number of classes on pretrained model, will default to 1000.")
        return model
    print(f"Passing {kwargs} args to torch to construct {model_arch}")
    return getattr(models, model_arch)(pretrained=pretrained, **kwargs)


def get_quantized_model(model_arch: str, kwargs={}):
    """
    Returns quantized version of the model from
    torchvision.models.quantization.
    If model is not supported, returns None.
    """
    model_arch = model_arch.lower()
    if hasattr(models.quantization, model_arch):
        model_params = getModelParams(model_arch)
        if "kwargs" in model_params:
            kwargs.update(model_params["kwargs"])
        print(f"Passing {kwargs} args to torch to construct {model_arch}")
        return getattr(models.quantization, model_arch)(**kwargs)
    
    print(f"Warning, model architecture {model_arch} is not supported for quanitzation, returning None from get_quantized_model().")
    return None


def fixLastLayer(model: torch.nn.Module, architecture: str, num_classes: int) -> bool:
    # TODO implement finetuning changing the last layer see here
    # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    supported_models = []   #["mnasnet1_3"]
    if architecture not in supported_models:
        return False
    return True
