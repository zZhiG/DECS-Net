import torch
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

def get_resnet18(is_bool):
    model = resnet18(pretrained=is_bool)
    return model

def get_resnet34(is_bool):
    model = resnet34(pretrained=is_bool)
    return model

def get_resnet50(is_bool):
    model = resnet50(pretrained=is_bool)
    return model

def get_resnet101(is_bool):
    model = resnet101(pretrained=is_bool)
    return model

def get_resnet152(is_bool):
    model = resnet152(pretrained=is_bool)
    return model

def get_resnet(model='resnet50', is_bool=True):
    if model == 'resnet18':
        return get_resnet18(is_bool)

    elif model == 'resnet50':
        return get_resnet50(is_bool)

    elif model == 'resnet34':
        return get_resnet34(is_bool)

    elif model == 'resnet101':
        return get_resnet101(is_bool)

    elif model == 'resnet152':
        return get_resnet152(is_bool)

    else:
        print('Not exist model!!!!')
        return ""

