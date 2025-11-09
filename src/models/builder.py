import os
from functools import partial
import timm

from models.ctran import ctranspath
from models.mils.model_clam import CLAM_MB, CLAM_SB
from models.mils.model_clam_enhanced import CLAM_MB_Enhanced, CLAM_SB_Enhanced
from models.mils.model_mil import MIL_fc, MIL_fc_mc
from models.mils.ABMIL import ABMIL
from models.mils.RRT import RRT
from models.mils.TransMIL import TransMIL
from models.mils.WiKG import WiKG
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


def build_mil_model(args, model_dict, device=None):
    """
    Unified MIL model builder supporting all techniques.

    Args:
        args: Arguments object with model configuration, OR string model_type for backward compatibility
        model_dict: Dictionary with model parameters (dropout, n_classes, embed_dim, etc.)
        device: Device for CUDA operations (optional, for backward compatibility)

    Returns:
        Initialized MIL model
    """
    # Backward compatibility: support old signature (model_type, model_dict, n_classes)
    if isinstance(args, str):
        model_type = args
        n_classes = device if device is not None else 2  # device was n_classes in old signature

        if model_type == 'clam_sb':
            model = CLAM_SB(**model_dict)
        elif model_type == 'clam_mb':
            model = CLAM_MB(**model_dict)
        else:  # model_type == 'mil'
            if n_classes > 2:
                model = MIL_fc_mc(**model_dict)
            else:
                model = MIL_fc(**model_dict)
        return model

    # New signature: full args object
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})

        if args.B > 0:
            model_dict.update({'k_sample': args.B})

        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device and device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()

        if args.model_type == 'clam_sb':
            if args.topo:
                model = CLAM_SB_Enhanced(**model_dict, instance_loss_fn=instance_loss_fn)
            else:
                model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            if args.topo:
                model = CLAM_MB_Enhanced(**model_dict, instance_loss_fn=instance_loss_fn)
            else:
                model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError

    elif args.model_type == 'abmil':
        # ABMIL: Attention-Based MIL
        abmil_dict = {
            'n_classes': args.n_classes,
            'in_dim': model_dict.get('embed_dim', 512),
            'hidden_dim': getattr(args, 'hidden_dim', 512),
            'dropout': model_dict.get('dropout', 0.3),
            'is_norm': getattr(args, 'abmil_is_norm', True)
        }
        model = ABMIL(**abmil_dict)

    elif args.model_type == 'rrt':
        # RRT: Region-Representative Transformer
        rrt_dict = {
            'n_classes': args.n_classes,
            'in_dim': model_dict.get('embed_dim', 512),
            'hidden_dim': getattr(args, 'hidden_dim', 512)
        }
        model = RRT(**rrt_dict)

    elif args.model_type == 'transmil':
        # TransMIL: Transformer MIL
        transmil_dict = {
            'n_classes': args.n_classes,
            'in_dim': model_dict.get('embed_dim', 512),
            'hidden_dim': getattr(args, 'hidden_dim', 512)
        }
        model = TransMIL(**transmil_dict)

    elif args.model_type == 'wikg':
        # WiKG: Weakly-supervised Instance-level Knowledge Graph
        wikg_dict = {
            'n_classes': args.n_classes,
            'in_dim': model_dict.get('embed_dim', 512),
            'hidden_dim': getattr(args, 'hidden_dim', 512)
        }
        model = WiKG(**wikg_dict)

    else:  # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    return model