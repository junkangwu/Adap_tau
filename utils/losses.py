import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class Adap_tau_Loss(nn.Module):
    def __init__(self):
        super(Adap_tau_Loss, self).__init__()

    def forward(self, y_pred, temperature_, w_0):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = torch.exp(y_pred[:, 0] *  w_0)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] * temperature_.unsqueeze(1))  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)

        Ng = neg_logits.sum(dim=-1)

        loss = (- torch.log(pos_logits / Ng))
        # log out
        pos_logits_ = torch.exp(y_pred[:, 0])  #  B
        neg_logits_ = torch.exp(y_pred[:, 1:])  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)

        Ng_ = neg_logits_.sum(dim=-1)

        loss_ = (- torch.log(pos_logits_ / Ng_)).detach()
        return loss, loss_

class SSM_Loss(nn.Module):
    def __init__(self, margin=0, temperature=1., negative_weight=None, pos_mode=None):
        super(SSM_Loss, self).__init__()
        self._margin = margin 
        self._temperature = temperature
        self._negative_weight = negative_weight
        self.pos_mode = pos_mode
        print("Here is SSM LOSS: tau is {}".format(self._temperature))

    def forward(self, y_pred):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = torch.exp(y_pred[:, 0] / self._temperature)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] / self._temperature)  # B M

        Ng = neg_logits.sum(dim=-1)

        loss = (- torch.log(pos_logits / Ng))
        
        return loss, 0.


