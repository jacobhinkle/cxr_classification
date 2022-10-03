import torch
from torch import nn

import mimic_cxr_jpg

def cxr_net(
        arch='densenet121',
        variant='plain',  # choices: plain, msd
        pretrained=False,
        num_classes=len(mimic_cxr_jpg.chexpert_labels),
    ):
    if 'densenet' in arch:
        from torchvision.models import densenet

        if arch == 'densenet121':
            c = densenet.densenet121
            num_init_features = 64
        elif arch == 'densenet161':
            c = densenet.densenet161
            num_init_features = 96
        elif arch == 'densenet169':
            c = densenet.densenet169
            num_init_features = 64
        elif arch == 'densenet201':
            c = densenet.densenet201
            num_init_features = 64
        else:
            raise ValueError('arch must be one of: densenet121, densenet161, densenet169, densenet201')

        mod = c(pretrained=pretrained, num_classes=1000)
        # modify first conv to take proper input_channels
        oldconv = mod.features.conv0
        newconv = nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        newconv.weight.data = oldconv.weight.data.sum(dim=1, keepdims=True)
        mod.features._modules['conv0'] = newconv
        mod.classifier = nn.Linear(mod.classifier.in_features, num_classes)

        match variant:
            case 'msd':
                # convert each block to msd style
                submods = dict(mod.named_modules())
                for i in range(4):
                    db = submods[f'features.denseblock{i+1}']
                    blockmods = dict(db.named_modules())
                    l = 1
                    while f'denselayer{l}' in blockmods:
                        layer = blockmods[f'denselayer{l}']
                        layermods = dict(layer.named_modules())
                        # conv1 is a 1x1 that goes to bottleneck. conv2 is a 3x3
                        layermods['conv2'].dilation = l
                        layermods['conv2'].padding = l
                        print(f'Block {i} Layer {l} conv2.dilation={l}')
                        l += 1


    elif 'resnet' in arch:
        from torchvision.models import resnet
        if arch == 'resnet50':
            c = resnet.resnet50(pretrained=True,replace_stride_with_dilation=[False, True, True])
            num_init_features = 64
        elif arch == 'resnet101':
            c = resnet.resnet101
            num_init_features = 64
        else:
            raise ValueError('arch must be one of: resnet50, resnet101')

        mod = c
        # modify first conv to take proper input_channels
        oldconv = mod.conv1
        newconv = nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        newconv.weight.data = oldconv.weight.data.sum(dim=1, keepdims=True)
        mod.conv1 = newconv
        mod.fc = nn.Linear(512 * 4, num_classes)
    return mod


def cxr_feature_net(*args, **kwargs):
    return cxr_net(*args, **kwargs).features


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arch',
        default='densenet121',
        choices=('densenet121', ),
        help='Model architecture',
    )
    parser.add_argument(
        '--variant',
        default='plain',
        choices=('plain', 'msd'),
        help='Variant of architecture',
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        help='Whether to download pretrained weights using torchvision',
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=14,
        help='Number of scalar outputs',
    )
    args = parser.parse_args()

    model = cxr_net(**vars(args))
    try:
        import torchinfo
        torchinfo.summary(model)
    except ImportError:
        import warnings
        warnings.warn("More info is available if you install torchinfo")

    def count_params(m):
        n = 0
        for p in m.parameters():
            n += p.numel()
        return n

    nfeat = count_params(model.features)
    nclass = count_params(model.classifier)
    print("Num parameters:")
    print("  Feature network:", nfeat)
    print("  Classifier network:", nclass)
    print("  Total:", nclass + nfeat)

