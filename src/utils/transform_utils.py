from torchvision import transforms


class AutoImageProcessorCompose(transforms.Compose):
    def __init__(self, auto_image_processor):
        self.transforms = auto_image_processor

    def __call__(self, img):
        return self.transforms(img, return_tensors = 'pt')

    def __repr__(self) -> str:
        return str(self.transforms)

def get_eval_musk_transforms(mean, std, target_img_size = -1):
	trsforms = []
	trsforms.append(transforms.Resize(384, interpolation=3, antialias=True))
	trsforms.append(transforms.CenterCrop((384, 384)))
	trsforms.append(transforms.ToTensor())
	trsforms.append(transforms.Normalize(mean, std))
	trsforms = transforms.Compose(trsforms)
	return trsforms

def get_eval_gigapath_transforms(mean, std, target_img_size = -1):
	trsforms = []
	trsforms.append(transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC))
	transforms.CenterCrop(224), 
	trsforms.append(transforms.Resize(224))   
	trsforms.append(transforms.ToTensor())
	trsforms.append(transforms.Normalize(mean, std))
	trsforms = transforms.Compose(trsforms)

	return trsforms

def get_eval_transforms(mean, std, target_img_size = -1):
	trsforms = []
	
	if target_img_size > 0:
		trsforms.append(transforms.Resize(target_img_size))
	trsforms.append(transforms.ToTensor())
	trsforms.append(transforms.Normalize(mean, std))
	trsforms = transforms.Compose(trsforms)

	return trsforms