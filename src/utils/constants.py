from timm.data.constants import *

MODEL2CONSTANTS = {
	"resnet50_trunc": {
		"mean": IMAGENET_DEFAULT_MEAN,
		"std": IMAGENET_DEFAULT_STD
	},
	"uni_v1":
	{
		"mean": IMAGENET_DEFAULT_MEAN,
		"std": IMAGENET_DEFAULT_STD
	},
    "uni_v2":
	{
		"mean": IMAGENET_DEFAULT_MEAN,
		"std": IMAGENET_DEFAULT_STD
	},
	"conch_v1":
	{
		"mean": OPENAI_CLIP_MEAN,
		"std": OPENAI_CLIP_STD
	},
    "ctranspath":
    {
		"mean": IMAGENET_DEFAULT_MEAN,
		"std": IMAGENET_DEFAULT_STD
	},
    "retccl":
    {
		"mean": IMAGENET_DEFAULT_MEAN,
		"std": IMAGENET_DEFAULT_STD
	}, 
    "provgigapath":
	{
		"mean": IMAGENET_DEFAULT_MEAN,
		"std": IMAGENET_DEFAULT_STD
	},
    "musk":{
		"mean": IMAGENET_INCEPTION_MEAN,
		"std": IMAGENET_INCEPTION_STD
	},'hoptimus0':{	
        "mean": (0.707223, 0.578729, 0.703617),
		"std": (0.211883, 0.230117, 0.177517)
	}
}