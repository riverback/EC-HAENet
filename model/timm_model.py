import os
import torch
import torchvision
from torchvision import transforms
import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform


model_name_list = [
    'resnet18.a1_in1k',
    'resnet50.a1_in1k',
    'resnet101.a1h_in1k',
    'densenet121.ra_in1k',
    'convnext_base.fb_in1k',
    'inception_v3.tv_in1k',
    'xception41.tf_in1k',
    'vit_base_patch16_224.augreg_in1k',
    'swin_base_patch4_window7_224.ms_in1k',
    'mobilenetv2_100.ra_in1k',
    'mobilenetv3_large_100.ra_in1k',
    'efficientvit_b0.r224_in1k'
]

def get_local_timm_checkpoint_path(model_name):
    ckpt_folder = os.path.join('model/pretrain_checkpoints/hub',
                                       'models--timm--'+model_name, 'snapshots')
    ckpt_path = None
    # find the safetensors path with os.walk
    for root, dirs, files in os.walk(ckpt_folder):
        for name in files:
            if name.endswith('.safetensors'):
                ckpt_path = os.path.join(root, name)
                break
    return ckpt_path

def get_timm_model(model_name, num_classes, pretrained=True, checkpoint_path=None, args=None) -> torch.nn.Module:
    ''' return model, config, transform
    if want to use the pretrained weight, set pretrained=False, checkpoint_path=path_to_weight,
    or set pretrained=True, checkpoint_path=None.
    if want the model to be randomly initialized, set pretrained=False, checkpoint_path=None.
    '''

    if checkpoint_path is not None:
        assert os.path.exists(checkpoint_path), 'checkpoint_path not exists'
        # assert checkpoint_path.endswith('.safetensors'), 'checkpoint_path for timm model should be a .safetensors file'
    
    # some bug in swin weight
    if 'swin' in model_name and checkpoint_path is not None:
        checkpoint_path = None
        pretrained = True
    try:
        model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)
    except:
        print('Failed to load model from local, try to load from hf-mirror.com')
        os.system('export HF_ENDPOINT=https://hf-mirror.com')
        model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)

    model.reset_classifier(num_classes=num_classes)
    if pretrained and checkpoint_path is None:
        checkpoint_path = 'huggingface'
    print(f'Load timm model {model_name} with {num_classes} classes, weight from {checkpoint_path}')
    config = resolve_model_data_config(model, None)
    # transform = create_transform(**config) # don't want normalization here
    if args is not None:
        input_size = (args.input_size, args.input_size)
    else:
        input_size = config['input_size'][-2:]
    transform = transforms.Compose([
        transforms.Resize(size=input_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std']),
    ])
    return model, config, transform

def get_trained_model(model_name, num_classes, checkpoint_path):
    '''
    use our trained model weight to initialize the model
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    '''
    # checkpoint_path: weight.pth
    assert os.path.exists(checkpoint_path), 'checkpoint_path not exists'
    assert checkpoint_path.endswith('.pth'), 'checkpoint_path for our trained model should be a .pth file'
    model = timm.create_model(model_name, pretrained=False)
    model.reset_classifier(num_classes=num_classes)
    # load the trained model
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['model'])
    print(f'Load {model_name} weights from {checkpoint_path}')
    return model

def get_ood_model(model_name, num_classes, shared_checkpoint_path):
    assert os.path.exists(shared_checkpoint_path), 'shared_checkpoint_path not exists'
    encoder = timm.create_model(model_name, pretrained=False)
    encoder.reset_classifier(num_classes=num_classes)
    try:
        state = torch.load(shared_checkpoint_path, map_location='cpu')
        encoder.load_state_dict(state['model'])
    except:
        raise ValueError('Failed to load shared model weight')
    
    # return the feature extractor
    encoder.reset_classifier(num_classes=0)
    fc = torch.nn.Sequential(
        torch.nn.Linear(encoder.num_features, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, num_classes+1)
    )
    ood_model = torch.nn.Sequential(
        encoder, 
        fc
    )
    
    return ood_model

def get_ood_classifier(in_dim, num_classes):
    fc = torch.nn.Sequential(
        torch.nn.Linear(in_dim, in_dim // 2),
        torch.nn.ReLU(),
        torch.nn.Linear(in_dim // 2, num_classes+1)
    )
    return fc