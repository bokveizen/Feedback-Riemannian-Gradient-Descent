import torch
from torch.optim.optimizer import Optimizer, required


def projection_ob_extended(p):
    # p[1] = F_{p[0]}(p[1])
    m, n = p[0].shape
    p[1].sub_(torch.bmm(p[1].view(m, 1, n), p[0].view(m, n, 1)).flatten()
              .div(p[0].norm(dim=1).pow(2))
              .unsqueeze(-1)
              .mul(p[0]))


def projection_st_extended(p):
    # p[1] = F_{p[0]}(p[1])
    m, n = p[0].shape
    eye_p = torch.eye(m, device=p[0].device)
    q = p[1].mm(p[0].t())
    q.add_(q.t()).mul_(0.5)
    inverse_approx = eye_p.mul(2).sub(p[0].mm(p[0].t()))
    p[1].sub_(q.mm(inverse_approx).mm(p[0]))


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

    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 feedback=0.01, punishment=0, conv_only=False):
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
        if conv_only:
            self.dim_list_to_process = [4]
        else:
            self.dim_list_to_process = [2, 4]

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
        dim_list_to_process = self.dim_list_to_process
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if p.dim() not in dim_list_to_process:  # original procedure
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
                else:
                    # no weight decay
                    # compute momentum first, then do Riemannian gradient and feedback operation
                    if p.dim() == 2:  # FC weight
                        eye_p = torch.eye(p.shape[0], device=p.device)
                        if p.shape[0] <= p.shape[1]:
                            if momentum != 0:  # Riemannian momentum
                                param_state = self.state[p]
                                if 'momentum_buffer' not in param_state:  # v0
                                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                                    buf.add_(d_p)
                                    projection_st_extended([p.data, buf])
                                else:
                                    buf = param_state['momentum_buffer']
                                    con = torch.zeros_like(p.data)
                                    inverse_approx = eye_p.mul(2).sub(p.data.mm(p.data.t()))
                                    con.add_(group['lr'], buf.mm(buf.t()).mm(inverse_approx).mm(p.data))
                                    rd = torch.zeros_like(p.data)
                                    rd.add_(momentum - 1, buf).add_(d_p)
                                    projection_st_extended([p.data, rd])
                                    buf.add_(con).add_(rd)
                                    projection_st_extended([p.data, buf])
                                if nesterov:  # TODO: Riemannian nesterov momentum
                                    d_p = d_p.add(momentum, buf)
                                else:
                                    d_p = buf
                            if self.feedback > 0:
                                # d_p.add_(self.feedback, 4 * p.data.mm(p.data.t()).sub(eye_p).mm(p.data))
                                d_p.add_(self.feedback, 4 * p.data.mm(p.data.t()).mm(p.data).sub(p.data))
                        else:  # Oblique
                            if momentum != 0:  # Riemannian momentum
                                param_state = self.state[p]
                                if 'momentum_buffer' not in param_state:  # v0
                                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                                    buf.add_(d_p)
                                    projection_ob_extended([p.data, buf])
                                else:
                                    buf = param_state['momentum_buffer']
                                    con = torch.zeros_like(p.data)
                                    con.add_(group['lr'],
                                             buf.norm(dim=1)
                                             .div(p.data.norm(dim=1))
                                             .pow(2)
                                             .unsqueeze(-1)
                                             .mul(p.data))
                                    rd = torch.zeros_like(p.data)
                                    rd.add_(momentum - 1, buf).add_(d_p)
                                    projection_ob_extended([p.data, rd])
                                    buf.add_(con).add_(rd)
                                    projection_ob_extended([p.data, buf])
                                if nesterov:  # TODO: Riemannian nesterov momentum
                                    d_p = d_p.add(momentum, buf)
                                else:
                                    d_p = buf
                            if self.feedback > 0:
                                # d_p.add_(self.feedback, 4 * p.data.mm(p.data.t()).sub(eye_p).diag().diag().mm(p.data))
                                d_p.add_(self.feedback, 4 * p.data.norm(dim=1).pow(2).sub(1).unsqueeze(-1).mul(p.data))
                    elif p.dim() == 4:  # Conv weight
                        p_2d = p.data.view(p.shape[0], -1)
                        # d_p_2d = d_p.view_as(p_2d)
                        eye_p_2d = torch.eye(p_2d.shape[0], device=p.device)
                        if p_2d.shape[0] <= p_2d.shape[1]:
                            if momentum != 0:  # Riemannian momentum
                                param_state = self.state[p]
                                if 'momentum_buffer' not in param_state:  # v0
                                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                                    buf.add_(d_p)
                                    buf_2d = buf.view_as(p_2d)
                                    projection_st_extended([p_2d, buf_2d])
                                else:
                                    buf = param_state['momentum_buffer']
                                    buf_2d = buf.view_as(p_2d)
                                    con = torch.zeros_like(p.data)
                                    inverse_approx = eye_p_2d.mul(2).sub(p_2d.mm(p_2d.t()))
                                    con.add_(group['lr'],
                                             buf_2d.mm(buf_2d.t()).mm(inverse_approx).mm(p_2d).view_as(d_p))
                                    rd = torch.zeros_like(p.data)
                                    rd.add_(momentum - 1, buf).add_(d_p)
                                    rd_2d = rd.view_as(p_2d)
                                    projection_st_extended([p_2d, rd_2d])
                                    buf.add_(con).add_(rd)
                                    projection_st_extended([p_2d, buf_2d])
                                if nesterov:  # TODO: Riemannian nesterov momentum
                                    d_p = d_p.add(momentum, buf)
                                else:
                                    d_p = buf
                            if self.feedback > 0:
                                d_p.add_(self.feedback, 4 * p_2d.mm(p_2d.t()).mm(p_2d).sub(p_2d).view_as(d_p))
                        else:  # Oblique
                            p_2d = p.data.view(p.shape[0], -1)
                            if momentum != 0:  # Riemannian momentum
                                param_state = self.state[p]
                                if 'momentum_buffer' not in param_state:  # v0
                                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                                    buf.add_(d_p)
                                    buf_2d = buf.view_as(p_2d)
                                    projection_ob_extended([p_2d, buf_2d])
                                else:
                                    buf = param_state['momentum_buffer']
                                    buf_2d = buf.view_as(p_2d)
                                    con = torch.zeros_like(p.data)
                                    con.add_(group['lr'],
                                             buf_2d.norm(dim=1)
                                             .div(p_2d.norm(dim=1))
                                             .pow(2)
                                             .unsqueeze(-1)
                                             .mul(p_2d)
                                             .view_as(d_p))
                                    rd = torch.zeros_like(p.data)
                                    rd.add_(momentum - 1, buf).add_(d_p)
                                    rd_2d = rd.view_as(p_2d)
                                    projection_ob_extended([p_2d, rd_2d])
                                    buf.add_(con).add_(rd)
                                    projection_ob_extended([p_2d, buf_2d])
                                if nesterov:  # TODO: Riemannian nesterov momentum
                                    d_p = d_p.add(momentum, buf)
                                else:
                                    d_p = buf
                            if self.feedback > 0:
                                d_p.add_(self.feedback,
                                         4 * p_2d.norm(dim=1).pow(2).sub(1).unsqueeze(-1).mul(p_2d).view_as(d_p))
                p.data.add_(-group['lr'], d_p)
        return loss
