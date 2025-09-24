from bcistainer.evaluate import BCIEvaluatorBasic, BCIEvaluatorCAHR

import torch
import numpy as np

from bcistainer.utils import normalize_image, unnormalize_image, tta, untta


class BCIEvaluatorCAHRExt(BCIEvaluatorCAHR):
    def __init__(self, configs, model_path, apply_tta=False):
        super().__init__(configs, model_path, apply_tta)

    @torch.no_grad()
    def predict(self, he_ori):
        he = normalize_image(he_ori, 'he', self.norm_method)
        he = he.transpose(2, 0, 1).astype(np.float32)[None, ...]
        he = torch.Tensor(he).to(self.device)

        he_crop = self._crop(he_ori)
        he_crop = normalize_image(he_crop, 'he', self.norm_method)
        he_crop = he_crop.transpose(0, 3, 1, 2).astype(np.float32)
        he_crop = torch.Tensor(he_crop).to(self.device)

        multi_outputs = self.G(he, he_crop, self.crop_idxs, self.infer_mode)
        ihc_pred = multi_outputs[0]
        ihc_pred = ihc_pred[0].cpu().numpy()
        ihc_pred = ihc_pred.transpose(1, 2, 0)
        ihc_pred = unnormalize_image(ihc_pred, 'ihc', self.norm_method)
        ihc_pred = ihc_pred.astype(np.uint8)

        return ihc_pred

    @torch.no_grad()
    def predict_tta(self, he_ori):
        ihc_pred_tta = np.zeros_like(he_ori).astype(np.float32)
        for i in range(7):
            he_tta = tta(he_ori, i)
            he = normalize_image(he_tta, 'he', self.norm_method)
            he = he.transpose(2, 0, 1).astype(np.float32)[None, ...]
            he = torch.Tensor(he).to(self.device)

            he_crop = self._crop(he_tta)
            he_crop = normalize_image(he_crop, 'he', self.norm_method)
            he_crop = he_crop.transpose(0, 3, 1, 2).astype(np.float32)
            he_crop = torch.Tensor(he_crop).to(self.device)

            multi_outputs = self.G(he, he_crop, self.crop_idxs, self.infer_mode)
            ihc_pred = multi_outputs[0]
            ihc_pred = ihc_pred[0].cpu().numpy()
            ihc_pred = ihc_pred.transpose(1, 2, 0)
            ihc_pred = unnormalize_image(ihc_pred, 'ihc', self.norm_method)

            ihe_pred_untta = untta(ihc_pred, i)
            ihc_pred_tta += ihe_pred_untta

        ihc_pred_tta /= 7
        ihc_pred_tta = ihc_pred_tta.astype(np.uint8)

        return ihc_pred_tta

class BCIEvaluatorBasicExt(BCIEvaluatorBasic):

    def __init__(self, configs, model_path, apply_tta=False):
        super().__init__(configs, model_path, apply_tta)
        
    @torch.no_grad()
    def predict(self, he_ori):
        he = normalize_image(he_ori, 'he', self.norm_method)
        he = he.transpose(2, 0, 1).astype(np.float32)[None, ...]
        he = torch.Tensor(he).to(self.device)

        multi_outputs = self.G(he)
        ihc_pred = multi_outputs[0]
        ihc_pred = ihc_pred[0].cpu().numpy()
        ihc_pred = ihc_pred.transpose(1, 2, 0)
        ihc_pred = unnormalize_image(ihc_pred, 'ihc', self.norm_method)
        ihc_pred = ihc_pred.astype(np.uint8)


        return ihc_pred
    
    @torch.no_grad()
    def predict_tta(self, he_ori):
        ihc_pred_tta = np.zeros_like(he_ori).astype(np.float32)
        for i in range(7):
            he_tta = tta(he_ori, i)
            he = normalize_image(he_tta, 'he', self.norm_method)
            he = he.transpose(2, 0, 1).astype(np.float32)[None, ...]
            he = torch.Tensor(he).to(self.device)

            multi_outputs = self.G(he)
            ihc_pred = multi_outputs[0]
            ihc_pred = ihc_pred[0].cpu().numpy()
            ihc_pred = ihc_pred.transpose(1, 2, 0)
            ihc_pred = unnormalize_image(ihc_pred, 'ihc', self.norm_method)

            ihe_pred_untta = untta(ihc_pred, i)
            ihc_pred_tta += ihe_pred_untta

        ihc_pred_tta /= 7
        ihc_pred_tta = ihc_pred_tta.astype(np.uint8)
    
        return ihc_pred_tta
