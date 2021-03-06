import torch
from torch.optim.optimizer import Optimizer, required


class SGDOblique(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = SGDOblique(model.parameters(), lr=0.1, momentum=0.9)
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

    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 oblique=False, feedback=(1, 1)):
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
        super(SGDOblique, self).__init__(params, defaults)
        self.feedback = feedback
        self.oblique = oblique

    def __setstate__(self, state):
        super(SGDOblique, self).__setstate__(state)
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
        # dim_list_to_process = self.dim_list_to_process
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if not self.oblique:  # original procedure
                    if weight_decay != 0:
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
                else:  # oblique
                    # no weight decay
                    if p.dim() == 2:  # FC weight
                        if momentum != 0:  # Riemannian momentum
                            param_state = self.state[p]
                            if 'momentum_buffer' not in param_state:  # v0
                                buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                                buf.add_(d_p)
                                buf.sub_(torch.bmm(buf.unsqueeze(1), p.data.unsqueeze(2)).flatten()
                                         .div(p.data.norm(dim=1).pow(2))
                                         .unsqueeze(-1)
                                         .mul(p.data))
                                buf.mul_(-group['lr'])
                            else:
                                buf = param_state['momentum_buffer']

                                con = torch.zeros_like(p.data)
                                con.sub_(buf.norm(dim=1)
                                         .div(p.data.norm(dim=1))
                                         .pow(2)
                                         .unsqueeze(-1)
                                         .mul(p.data))

                                rd = torch.zeros_like(p.data)
                                rd.add_(momentum - 1, buf).add_(-group['lr'], d_p)
                                rd.sub_(torch.bmm(rd.unsqueeze(1), p.data.unsqueeze(2)).flatten()
                                        .div(p.data.norm(dim=1).pow(2))
                                        .unsqueeze(-1)
                                        .mul(p.data))

                                ext = torch.zeros_like(p.data)
                                ext.add_(torch.bmm(buf.unsqueeze(1), p.data.unsqueeze(2)).flatten()
                                         .div(p.data.norm(dim=1).pow(2))
                                         .unsqueeze(-1)
                                         .mul(buf))

                                fb = torch.zeros_like(p.data)
                                if self.feedback[1] > 0:  # alpha 2
                                    fb.sub_(self.feedback[1] * group['lr'],
                                            torch.bmm(buf.unsqueeze(1), p.data.unsqueeze(2)).flatten()
                                            .unsqueeze(-1)
                                            .mul(p.data))

                                buf.add_(con).add_(rd).add_(ext).add_(fb)
                            if nesterov:  # TODO: Riemannian nesterov momentum
                                d_p = d_p.add(momentum, buf)
                            else:
                                d_p = buf

                            rd = torch.zeros_like(p.data)
                            rd.sub_(torch.bmm(d_p.unsqueeze(1), p.data.unsqueeze(2)).flatten()
                                    .div(p.data.norm(dim=1).pow(2))
                                    .unsqueeze(-1)
                                    .mul(p.data))

                            fb = torch.zeros_like(p.data)
                            if self.feedback[0] > 0:
                                fb.sub_(self.feedback[0] * group['lr'],
                                        p.data.norm(dim=1).pow(2).sub(1).unsqueeze(-1).mul(p.data))
                            if self.feedback[1] > 0:
                                fb.sub_(self.feedback[1] * group['lr'],
                                        torch.bmm(d_p.unsqueeze(1), p.data.unsqueeze(2)).flatten()
                                        .unsqueeze(-1)
                                        .mul(d_p))

                            p.data.add_(d_p).add_(rd).add_(fb)
                        else:  # no momentum
                            d_p.mul_(-group['lr'])
                            d_p.sub_(torch.bmm(d_p.unsqueeze(1), p.data.unsqueeze(2)).flatten()
                                     .div(p.data.norm(dim=1).pow(2))
                                     .unsqueeze(-1)
                                     .mul(p.data))
                            if self.feedback[0] > 0:
                                d_p.sub_(self.feedback[0] * group['lr'],
                                         p.data.norm(dim=1).pow(2).sub(1).unsqueeze(-1).mul(p.data))
                            p.data.add_(d_p)
                    elif p.dim() == 4:  # Conv weight
                        p_2d = p.data.view(p.shape[0], -1)
                        # eye_p_2d = torch.eye(p_2d.shape[0], device=p.device)
                        if momentum != 0:  # Riemannian momentum
                            param_state = self.state[p]
                            if 'momentum_buffer' not in param_state:  # v0
                                buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                                buf.add_(d_p)
                                buf_2d = buf.view_as(p_2d)
                                buf_2d.sub_(torch.bmm(buf_2d.unsqueeze(1), p_2d.unsqueeze(2)).flatten()
                                            .div(p_2d.norm(dim=1).pow(2))
                                            .unsqueeze(-1)
                                            .mul(p_2d))
                                buf.mul_(-group['lr'])
                            else:
                                buf = param_state['momentum_buffer']
                                buf_2d = buf.view_as(p_2d)
                                con = torch.zeros_like(p.data)
                                con_2d = con.view_as(p_2d)
                                con_2d.sub_(buf_2d.norm(dim=1)
                                            .div(p_2d.norm(dim=1))
                                            .pow(2)
                                            .unsqueeze(-1)
                                            .mul(p_2d))
                                rd = torch.zeros_like(p.data)
                                rd_2d = rd.view_as(p_2d)
                                rd.add_(momentum - 1, buf).add_(-group['lr'], d_p)
                                rd_2d.sub_(torch.bmm(rd_2d.unsqueeze(1), p_2d.unsqueeze(2)).flatten()
                                           .div(p_2d.norm(dim=1).pow(2))
                                           .unsqueeze(-1)
                                           .mul(p_2d))
                                ext = torch.zeros_like(p.data)
                                ext_2d = ext.view_as(p_2d)
                                ext_2d.add_(torch.bmm(buf_2d.unsqueeze(1), p_2d.unsqueeze(2)).flatten()
                                            .div(p_2d.norm(dim=1).pow(2))
                                            .unsqueeze(-1)
                                            .mul(buf_2d))
                                fb = torch.zeros_like(p.data)
                                fb_2d = fb.view_as(p_2d)
                                if self.feedback[1] > 0:  # alpha 2
                                    fb_2d.sub_(self.feedback[1] * group['lr'],
                                               torch.bmm(buf_2d.unsqueeze(1), p_2d.unsqueeze(2)).flatten()
                                               .unsqueeze(-1)
                                               .mul(p_2d))
                                buf.add_(con).add_(rd).add_(ext).add_(fb)
                            if nesterov:  # TODO: Riemannian nesterov momentum
                                d_p = d_p.add(momentum, buf)
                            else:
                                d_p = buf

                            d_p_2d = d_p.view_as(p_2d)
                            rd = torch.zeros_like(p.data)
                            rd_2d = rd.view_as(p_2d)
                            rd_2d.sub_(torch.bmm(d_p_2d.unsqueeze(1), p_2d.unsqueeze(2)).flatten()
                                       .div(p_2d.norm(dim=1).pow(2))
                                       .unsqueeze(-1)
                                       .mul(p_2d))

                            fb = torch.zeros_like(p.data)
                            fb_2d = fb.view_as(p_2d)
                            if self.feedback[0] > 0:
                                fb_2d.sub_(self.feedback[0] * group['lr'],
                                           p_2d.norm(dim=1).pow(2).sub(1).unsqueeze(-1).mul(p_2d))
                            if self.feedback[1] > 0:
                                fb_2d.sub_(self.feedback[1] * group['lr'],
                                           torch.bmm(d_p_2d.unsqueeze(1), p_2d.unsqueeze(2)).flatten()
                                           .unsqueeze(-1)
                                           .mul(d_p_2d))

                            p.data.add_(d_p).add_(rd).add_(fb)
                        else:  # no momentum
                            d_p.mul_(-group['lr'])
                            d_p_2d = d_p.view_as(p_2d)
                            d_p_2d.sub_(torch.bmm(d_p_2d.unsqueeze(1), p_2d.unsqueeze(2)).flatten()
                                        .div(p_2d.norm(dim=1).pow(2))
                                        .unsqueeze(-1)
                                        .mul(p_2d))
                            if self.feedback[0] > 0:
                                d_p_2d.sub_(self.feedback[0] * group['lr'],
                                            p_2d.norm(dim=1).pow(2).sub(1).unsqueeze(-1).mul(p_2d))
                            p.data.add_(d_p)
        return loss
