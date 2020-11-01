# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
from fairseq import utils
import numpy as np

from . import FairseqCriterion, register_criterion


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.cuda = torch.cuda.is_available() and not args.cpu
        if args.adaptive_training:
            self.weight_drop = args.weight_drop
            freq = []
            for x, y in read2columns(args.dict_file, split=' '):
                freq.append(int(y))
            freq = torch.tensor(freq)
            mid = freq[int(len(freq) / 2)]
            if args.adaptive_method is 'exp':
                self.weight = [torch.exp(-1 * args.adaptive_T * item / mid) for item in freq]
                b = self.weight.max()
                self.weight = self.weight / b * (np.e - 1) + 1
            else:
                self.weight = [torch.pow(item / mid, torch.tensor(2)) * torch.exp(-1 * args.adaptive_T * item / mid) for item in freq]
                b = self.weight.max()
                self.weight = self.weight / b * (np.e - 1) + 1
            self.weight = torch.cat([torch.tensor([1., 1., 1., 1.]), self.weight], dim=0)
            if self.cuda:
                self.weight = self.weight.cuda()
            if args.fp16:
                self.weight = self.weight.half()



    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--adaptive-training', action='store_true',
                            help='if set, start token-level adaptive training.')
        parser.add_argument('--dict-file', default='dict.txt',
                            help='the target dictionary produced by fairseq itself.')
        parser.add_argument('--adaptive-method', default='exp', choices=['exp', 'k2'],
                            help='two methods mentioned in the paper')
        parser.add_argument('--adaptive-T', default=1., type=float,
                            help='The hyperparameter T.')
        parser.add_argument('--weight-drop', default=0.1, type=float,
                            help='A useful trick for adaptive training.')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        target = target.view(-1,1)
        non_pad_mask = target.ne(self.padding_idx)
        loss_weight = self.weight[target]
        drop_p = self.weight_drop * torch.ones_like(loss_weight)
        drop_mask = torch.bernoulli(drop_p).byte()
        loss_weight.masked_fill_(drop_mask, 1.)
        nll_loss = -(loss_weight * (lprobs.gather(dim=-1, index=target)))[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
