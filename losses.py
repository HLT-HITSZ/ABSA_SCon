from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def normalization(data):

    for i in range(len(data)):

        _range = torch.max(data[i]) - torch.min(data[i])
        data[i] = (data[i] - torch.min(data[i])) / _range
    return data


class SupConLoss(nn.Module):
    
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], features.shape[1], -1)

        # Get batch_size
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)     # 16*1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)      # 16*16
        else:
            mask = mask.float().to(device)

        features = features.unsqueeze(dim=1)
        features = F.normalize(features, dim=2)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            # Set contrast count
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # Get logits
        logits = anchor_dot_contrast - logits_max.detach()

        logits_min, _ = torch.min(logits, dim=1, keepdim=True)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        _range = logits_max - logits_min
        logits = torch.div(logits-logits_min,_range)

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print("mask",mask)  # 16*16

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), # Returns a tensor filled with the value 1
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # print("logits_mask",logits_mask)  #16*16
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1)

        # Get loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()


        return loss
