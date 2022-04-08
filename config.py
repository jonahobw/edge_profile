from get_model import name_to_family


VALID_MODELS = list(name_to_family.keys())

# models for which to collect profiles for
MODELS = VALID_MODELS   # MODELS = ["googlenet", "mobilenetv3", "resnet", "vgg"]
SYSTEM_SIGNALS = ["sm_clock_(mhz)", "memory_clock_(mhz)", "temperature_(c)", "power_(mw)", "fan_(%)"]
CHANNELS = 3
INPUT_SIZE = 224