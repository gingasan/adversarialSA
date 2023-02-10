import torch
import torch.nn as nn
from torch.autograd import Function


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


revgrad = RevGrad.apply


def gumbel_sigmoid_sub(x, tau=1e-12, training=True):
    if not training:
        return (x / tau).sigmoid()

    y = x.sigmoid()

    g1 = -torch.empty_like(x).exponential_().log()
    y_hard = ((x + g1 - g1) / tau).sigmoid()

    y_hard = (y_hard - y).detach() + y
    return y_hard


def divergence(query, target):
    if query.dim() <= 2 and query.shape[1] == 1:
        query = query.view(-1)
        target = target.view(-1)

    div_fct = nn.KLDivLoss(reduction="batchmean")
    q2t = div_fct(torch.log_softmax(query, dim=-1), torch.softmax(target, dim=-1))
    t2q = div_fct(torch.log_softmax(target, dim=-1), torch.softmax(query, dim=-1))
    return 1 / 2 * (q2t + t2q)
