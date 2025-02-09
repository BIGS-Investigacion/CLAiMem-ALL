import os
from functools import partial
import timm

from models.ctran import ctranspath
from models.musk import MUSKWrapper
from models.automodel_wrapper import AutoModelWrapper
from models.retccl import resnet50
from models.virchow import VirchowWrapper
from timm_1_0_14.timm.data.config import resolve_data_config
from timm_1_0_14.timm.data.transforms_factory import create_transform
from .timm_wrapper import TimmCNNEncoder

from timm_1_0_14 import timm as timm1014

from transformers import AutoImageProcessor, AutoModel

import torch
import torch.nn as nn

from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import AutoImageProcessorCompose, get_eval_gigapath_transforms, get_eval_musk_transforms, get_eval_transforms


def has_GENERIC():
    HAS_GENERIC = False
    GENERIC_CKPT_PATH = ''
    try:
        if 'GENERIC_CKPT_PATH' not in os.environ:
            raise ValueError('GENERIC_CKPT_PATH not set')
        HAS_GENERIC = True
        GENERIC_CKPT_PATH = os.environ['GENERIC_CKPT_PATH']
    except Exception as e:
        print(e)
        print('GENERIC not installed or GENERIC_CKPT_PATH not set')
    return HAS_GENERIC, GENERIC_CKPT_PATH

def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH

def get_encoder(model_name, target_img_size=224):
    #TODO: add support for other models

    img_transforms = None
    model = None
    
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_transforms(mean=constants['mean'],
                                            std=constants['std'],
                                            target_img_size = target_img_size)
    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_transforms(mean=constants['mean'],
                                            std=constants['std'],
                                            target_img_size = target_img_size)
    elif model_name == 'uni_v2':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        model = timm.create_model(
            pretrained=False, **timm_kwargs
        )
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_transforms(mean=constants['mean'],
                                            std=constants['std'],
                                            target_img_size = target_img_size)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_transforms(mean=constants['mean'],
                                            std=constants['std'],
                                            target_img_size = target_img_size)
    elif model_name == 'ctranspath':
        HAS_GENERIC, GENERIC_CKPT_PATH = has_GENERIC()
        assert HAS_GENERIC, 'CtransPath is not available'
        model = ctranspath()
        model.head = nn.Identity()
        model.load_state_dict(torch.load(GENERIC_CKPT_PATH, map_location="cpu")['model'], strict=True)
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_transforms(mean=constants['mean'],
                                            std=constants['std'],
                                            target_img_size = target_img_size)
    elif model_name == 'retccl':
        HAS_GENERIC, GENERIC_CKPT_PATH = has_GENERIC()
        assert HAS_GENERIC, 'RetCCL is not available'
        model = resnet50(num_classes=128,mlp=False, two_branch=False, normlinear=True)
        pretext_model = torch.load(GENERIC_CKPT_PATH)
        model.fc = nn.Identity()
        model.load_state_dict(pretext_model, strict=True)
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_transforms(mean=constants['mean'],
                                            std=constants['std'],
                                            target_img_size = target_img_size)
    elif model_name == 'provgigapath':
        model = timm1014.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_gigapath_transforms(mean=constants['mean'],
                                            std=constants['std'],
                                            target_img_size = target_img_size)
    elif model_name == 'phikon':
        model = AutoModelWrapper(AutoModel.from_pretrained("owkin/phikon-v2"))
        img_transforms = AutoImageProcessorCompose(AutoImageProcessor.from_pretrained("owkin/phikon-v2"))

    elif model_name == 'musk':
        model = MUSKWrapper()
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_musk_transforms(mean=constants['mean'],
                                            std=constants['std'],
                                            target_img_size = target_img_size)
    elif model_name == 'virchow':
        # need to specify MLP layer and activation function for proper init
        model = VirchowWrapper()
        sub_model = model.get_model()
        img_transforms = create_transform(**resolve_data_config(sub_model.pretrained_cfg, model=sub_model))
    elif model_name == 'hoptimus0':
        model = timm1014.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True,init_values=1e-5, dynamic_img_size=False)
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_transforms(mean=constants['mean'],
                                            std=constants['std'],
                                            target_img_size = target_img_size)
    elif model_name == 'hibou_l':
        model = AutoModelWrapper(AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True))
        img_transforms = AutoImageProcessorCompose(AutoImageProcessor.from_pretrained("histai/hibou-L", trust_remote_code=True))
    elif model_name == 'hibou_b':
        model = AutoModelWrapper(AutoModel.from_pretrained("histai/hibou-b", trust_remote_code=True))
        img_transforms = AutoImageProcessorCompose(AutoImageProcessor.from_pretrained("histai/hibou-b", trust_remote_code=True))
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))

    return model, img_transforms