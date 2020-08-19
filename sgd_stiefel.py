import torch
from torch.optim.optimizer import Optimizer, required


class SGDStiefel(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = SGDStiefel(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGDOblique with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, feedback=True, punishment=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDStiefel, self).__init__(params, defaults)
        self.feedback = feedback
        self.punishment = punishment

    def __setstate__(self, state):
        super(SGDStiefel, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if p.dim() == 4:  # Conv weight
                    p_2d = p.data.view(p.shape[0], -1)
                    if p_2d.shape[0] <= p_2d.shape[1]:
                        d_p.sub_(0.5 * (d_p.view_as(p_2d).mm(p_2d.t()) +
                                        d_p.view_as(p_2d).mm(p_2d.t()).t()).mm(p_2d).view_as(d_p))
                        if self.feedback:
                            fb_cor_term = 4 * p_2d.mm(p_2d.t()).sub(torch.eye(p_2d.shape[0], device=p.device)).mm(p_2d)
                            d_p.add_(fb_cor_term.view_as(d_p))
                    else:  # Oblique
                        d_p.sub_(d_p.view_as(p_2d).mm(p_2d.t()).diag().diag().mm(p_2d).view_as(d_p))
                        if self.feedback:
                            if not self.punishment:
                                fb_cor_term = 4 * p_2d.mm(p_2d.t()).sub(torch.eye(p_2d.shape[0], device=p.device)).diag().diag().mm(p_2d)
                                d_p.add_(fb_cor_term.view_as(d_p))
                            else:
                                public = p_2d.mm(p_2d.t()).sub(torch.eye(p_2d.shape[0], device=p.device))
                                st_fb = 4 * public.mm(p_2d)
                                ob_fb = 4 * public.diag().diag().mm(p_2d)
                                fb_cor_term = st_fb * self.punishment + ob_fb * (1. - self.punishment)
                                d_p.add_(fb_cor_term.view_as(d_p))
                elif p.dim() == 2:  # FC weight
                    if p.shape[0] <= p.shape[1]:
                        d_p.sub_(0.5 * (d_p.mm(p.data.t()) + d_p.mm(p.data.t()).t()).mm(p.data))
                        if self.feedback:
                            d_p.add_(4 * p.data.mm(p.data.t()).sub(torch.eye(p.shape[0], device=p.device)).mm(p.data))
                    else:  # Oblique
                        d_p.sub_(d_p.mm(p.data.t()).diag().diag().mm(p.data))
                        if self.feedback:
                            if not self.punishment:
                                d_p.add_(4 * p.data.mm(p.data.t()).sub(torch.eye(p.shape[0], device=p.device)).diag().diag().mm(p.data))
                            else:
                                public = p.data.mm(p.data.t()).sub(torch.eye(p.shape[0], device=p.device))
                                st_fb = 4 * public.mm(p.data)
                                ob_fb = 4 * public.diag().diag().mm(p.data)
                                ob_fb_w_punishment = st_fb * self.punishment + ob_fb * (1. - self.punishment)
                                d_p.add_(ob_fb_w_punishment)
                elif weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)

        return loss