import torch
#from torch.optim.optimizer import Optimizer, required
from Grad_optimizer import Optimizer, required

class Grad_SGD(Optimizer):
    """
    Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from:
        'On the importance of initialization and momentum in deep learning.'
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate.
        momentum (float, optional): momentum factor (default: 0).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        dampening (float, optional): dampening for momentum (default: 0).
        nesterov (bool, optional): enables Nesterov momentum (default: False).
    Example:
        >>> optimizer = Grad_SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    Considering the specific case of Momentum, the update can be written as:
        v = rho * v + g
        p = p - lr * v
    where p, g, v and rho denote the parameters, gradient, velocity, and momentum respectively.
    """
    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Grad_SGD, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(Grad_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
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
                d_p = p.grad.data   # TODO We need to smooth this by SJO operator
                
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
        
        return loss
