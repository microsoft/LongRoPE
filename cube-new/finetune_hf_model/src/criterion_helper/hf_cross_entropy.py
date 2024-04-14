import torch.nn.functional as F
from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterionConfig, CrossEntropyCriterion

IGNORE_IDX = -100


@register_criterion("hf_cross_entropy", dataclass=CrossEntropyCriterionConfig)
class HFCrossEntropyCriterion(CrossEntropyCriterion):
    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=IGNORE_IDX,
            reduction="sum" if reduce else "none",
        )
        return loss, loss
