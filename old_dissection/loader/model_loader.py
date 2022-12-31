import settings
import torch
import torchvision
import custom_model
from collections import OrderedDict


def vgg16_model(*args, **kwargs):

    # A version of vgg16 model where layers are given their research names: 
    model = torchvision.models.vgg16(*args, **kwargs)
    model.features = torch.nn.Sequential(OrderedDict(zip([
        'conv1_1', 'relu1_1',
        'conv1_2', 'relu1_2',
        'pool1',
        'conv2_1', 'relu2_1',
        'conv2_2', 'relu2_2',
        'pool2',
        'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2',
        'conv3_3', 'relu3_3',
        'pool3',
        'conv4_1', 'relu4_1',
        'conv4_2', 'relu4_2',
        'conv4_3', 'relu4_3',
        'pool4',
        'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2',
        'conv5_3', 'relu5_3',
        'pool5'],
        model.features)))

    model.classifier = torch.nn.Sequential(OrderedDict(zip([
        'fc6', 'relu6',
        'drop6',
        'fc7', 'relu7',
        'drop7',
        'fc8a'],
        model.classifier)))

    return model


def loadmodel(hook_fn):
    if settings.MODEL_FILE is None:
        model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
    else:
        checkpoint = torch.load(settings.MODEL_FILE)
        if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
            if settings.MODEL == 'custom':
                model = custom_model.ConvNet(num_classes=settings.NUM_CLASSES)
            elif settings.MODEL == 'vgg16':
                model = vgg16_model(num_classes=settings.NUM_CLASSES)
            else:
                model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
            
            if settings.MODEL_PARALLEL:
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                    'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
        else:
            model = checkpoint
    for name in settings.FEATURE_NAMES:
        #model._modules.get(name).register_forward_hook(hook_fn)
        layer = model._modules.get(name)
        if layer is None:
            for n,l in model.named_modules():
                print(n)
                if n == name:
                    layer = l
                    print('Layer found!')
                    break
        layer.register_forward_hook(hook_fn)
        
    if settings.GPU:
        model.cuda()
    model.eval()
    return model

