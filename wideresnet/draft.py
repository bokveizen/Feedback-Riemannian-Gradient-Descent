for key, value in params.items():
    if 'conv' in key and value.dim() == 4:  # conv weights
        param_g.append(value)
        key_g.append(key)
        if value.size()[0] <= np.prod(value.size()[1:]):  # stiefel
            q = qr_retraction(value.data.view(value.size(0), -1))
            value.data.copy_(q.view(value.size()))
        else:  # oblique
            unitp, _ = unit(value.data.view(value.size(0), -1))
            value.data.copy_(unitp.view(value.size()))
    elif 'bn' in key or 'bias' in key:
        param_e0.append(value)
    else:
        param_e1.append(value)

param_g = []
param_e = []
for key, value in net.state_dict().items():
    if value.dim() == 4:
        param_g.append(value)
        q = torch.zeros_like(value)
        value.data.copy_(q)
    else:
        param_e.append(value)
        q = torch.ones_like(value)
        value.data.copy_(q)
