import torch
from torch import nn
from scaled_for_pruning import epsilon
from scaled import LinearScaled
from scaled_for_pruning import Conv2dScaled, Conv2dScaledWithBN

# from scaled import *

# for m in net.modules():
# ...    if isinstance(m, Conv2dScaled):
# ...        active_scale = np.array([i for i in m.scale.data if i > 1e-6])
# ...        rsd = active_scale.std() / active_scale.mean()
# ...        print(rsd)


def para_counting_resnet18sp(net):
    res = 0
    current_c = (3, 3)

    # conv1
    scale = net.state_dict()['conv1.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv1.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # conv2_x
    # block 0
    scale = net.state_dict()['conv2_x.0.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.0.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv2_x.0.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.0.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 1
    scale = net.state_dict()['conv2_x.1.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.1.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv2_x.1.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.1.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # conv3_x
    # block 0 w/ shortcut
    # residual_function
    current_c_temp = current_c
    scale = net.state_dict()['conv3_x.0.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.0.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv3_x.0.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.0.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    res_last_scale_map = scale > epsilon
    # shortcut
    scale = net.state_dict()['conv3_x.0.shortcut.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.0.shortcut.0.conv.weight'].numel() \
           * current_c_temp[0] * remaining_output_c // (current_c_temp[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    sc_scale_map = scale > epsilon
    remaining_output_c = (res_last_scale_map + sc_scale_map > 0).sum().tolist()
    # for module w/ shortcut, the logical and of both maps should be kept for next module
    current_c = (remaining_output_c, full_output_c)

    # block 1
    scale = net.state_dict()['conv3_x.1.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.1.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv3_x.1.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.1.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # conv4_x
    # block 0 w/ short cut
    # residual_function
    current_c_temp = current_c
    scale = net.state_dict()['conv4_x.0.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.0.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv4_x.0.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.0.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    res_last_scale_map = scale > epsilon
    # shortcut
    scale = net.state_dict()['conv4_x.0.shortcut.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.0.shortcut.0.conv.weight'].numel() \
           * current_c_temp[0] * remaining_output_c // (current_c_temp[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    sc_scale_map = scale > epsilon
    remaining_output_c = (res_last_scale_map + sc_scale_map > 0).sum().tolist()
    # for module w/ shortcut, the logical and of both maps should be kept for next module
    current_c = (remaining_output_c, full_output_c)

    # block 1
    scale = net.state_dict()['conv4_x.1.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.1.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv4_x.1.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.1.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # conv5_x
    # block 0 w/ short cut
    # residual_function
    current_c_temp = current_c
    scale = net.state_dict()['conv5_x.0.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.0.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv5_x.0.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.0.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    res_last_scale_map = scale > epsilon
    # shortcut
    scale = net.state_dict()['conv5_x.0.shortcut.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.0.shortcut.0.conv.weight'].numel() \
           * current_c_temp[0] * remaining_output_c // (current_c_temp[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    sc_scale_map = scale > epsilon
    remaining_output_c = (res_last_scale_map + sc_scale_map > 0).sum().tolist()
    # for module w/ shortcut, the logical and of both maps should be kept for next module
    current_c = (remaining_output_c, full_output_c)

    # block 1
    scale = net.state_dict()['conv5_x.1.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.1.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv5_x.1.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.1.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # fc
    full_para_num = net.state_dict()['fc.fc.weight'].numel() + net.state_dict()['fc.fc.bias'].numel()
    res += full_para_num * current_c[0] // current_c[1]

    return res


def para_counting_resnet18nbsp(net):
    res = 0
    current_c = (3, 3)

    # conv1
    scale = net.state_dict()['conv1.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv1.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    current_c = (remaining_output_c, full_output_c)

    # conv2_x
    # block 0
    scale = net.state_dict()['conv2_x.0.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.0.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv2_x.0.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.0.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    current_c = (remaining_output_c, full_output_c)

    # block 1
    scale = net.state_dict()['conv2_x.1.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.1.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv2_x.1.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.1.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    current_c = (remaining_output_c, full_output_c)

    # conv3_x
    # block 0 w/ shortcut
    # residual_function
    current_c_temp = current_c
    scale = net.state_dict()['conv3_x.0.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.0.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv3_x.0.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.0.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res_last_scale_map = scale > epsilon
    # shortcut
    scale = net.state_dict()['conv3_x.0.shortcut.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.0.shortcut.0.conv.weight'].numel() \
           * current_c_temp[0] * remaining_output_c // (current_c_temp[1] * full_output_c)
    sc_scale_map = scale > epsilon
    remaining_output_c = (res_last_scale_map + sc_scale_map > 0).sum().tolist()
    # for module w/ shortcut, the logical and of both maps should be kept for next module
    current_c = (remaining_output_c, full_output_c)

    # block 1
    scale = net.state_dict()['conv3_x.1.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.1.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv3_x.1.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.1.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    current_c = (remaining_output_c, full_output_c)

    # conv4_x
    # block 0 w/ short cut
    # residual_function
    current_c_temp = current_c
    scale = net.state_dict()['conv4_x.0.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.0.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv4_x.0.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.0.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res_last_scale_map = scale > epsilon
    # shortcut
    scale = net.state_dict()['conv4_x.0.shortcut.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.0.shortcut.0.conv.weight'].numel() \
           * current_c_temp[0] * remaining_output_c // (current_c_temp[1] * full_output_c)
    sc_scale_map = scale > epsilon
    remaining_output_c = (res_last_scale_map + sc_scale_map > 0).sum().tolist()
    # for module w/ shortcut, the logical and of both maps should be kept for next module
    current_c = (remaining_output_c, full_output_c)

    # block 1
    scale = net.state_dict()['conv4_x.1.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.1.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv4_x.1.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.1.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    current_c = (remaining_output_c, full_output_c)

    # conv5_x
    # block 0 w/ short cut
    # residual_function
    current_c_temp = current_c
    scale = net.state_dict()['conv5_x.0.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.0.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv5_x.0.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.0.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res_last_scale_map = scale > epsilon
    # shortcut
    scale = net.state_dict()['conv5_x.0.shortcut.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.0.shortcut.0.conv.weight'].numel() \
           * current_c_temp[0] * remaining_output_c // (current_c_temp[1] * full_output_c)
    sc_scale_map = scale > epsilon
    remaining_output_c = (res_last_scale_map + sc_scale_map > 0).sum().tolist()
    # for module w/ shortcut, the logical and of both maps should be kept for next module
    current_c = (remaining_output_c, full_output_c)

    # block 1
    scale = net.state_dict()['conv5_x.1.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.1.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv5_x.1.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.1.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    current_c = (remaining_output_c, full_output_c)

    # fc
    full_para_num = net.state_dict()['fc.fc.weight'].numel() + net.state_dict()['fc.fc.bias'].numel()
    res += full_para_num * current_c[0] // current_c[1]

    return res


def para_counting_resnet34sp(net):
    res = 0
    current_c = (3, 3)

    # conv1
    scale = net.state_dict()['conv1.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv1.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # conv2_x
    # block 0
    scale = net.state_dict()['conv2_x.0.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.0.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv2_x.0.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.0.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 1
    scale = net.state_dict()['conv2_x.1.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.1.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv2_x.1.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.1.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 2
    scale = net.state_dict()['conv2_x.2.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.2.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv2_x.2.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.2.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # conv3_x
    # block 0 w/ shortcut
    # residual_function
    current_c_temp = current_c
    scale = net.state_dict()['conv3_x.0.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.0.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv3_x.0.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.0.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    res_last_scale_map = scale > epsilon
    # shortcut
    scale = net.state_dict()['conv3_x.0.shortcut.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.0.shortcut.0.conv.weight'].numel() \
           * current_c_temp[0] * remaining_output_c // (current_c_temp[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    sc_scale_map = scale > epsilon
    remaining_output_c = (res_last_scale_map + sc_scale_map > 0).sum().tolist()
    # for module w/ shortcut, the logical and of both maps should be kept for next module
    current_c = (remaining_output_c, full_output_c)

    # block 1
    scale = net.state_dict()['conv3_x.1.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.1.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv3_x.1.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.1.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 2
    scale = net.state_dict()['conv3_x.2.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.2.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv3_x.2.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.2.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 3
    scale = net.state_dict()['conv3_x.3.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.3.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv3_x.3.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.3.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # conv4_x
    # block 0 w/ short cut
    # residual_function
    current_c_temp = current_c
    scale = net.state_dict()['conv4_x.0.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.0.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv4_x.0.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.0.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    res_last_scale_map = scale > epsilon
    # shortcut
    scale = net.state_dict()['conv4_x.0.shortcut.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.0.shortcut.0.conv.weight'].numel() \
           * current_c_temp[0] * remaining_output_c // (current_c_temp[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    sc_scale_map = scale > epsilon
    remaining_output_c = (res_last_scale_map + sc_scale_map > 0).sum().tolist()
    # for module w/ shortcut, the logical and of both maps should be kept for next module
    current_c = (remaining_output_c, full_output_c)

    # block 1
    scale = net.state_dict()['conv4_x.1.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.1.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv4_x.1.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.1.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 2
    scale = net.state_dict()['conv4_x.2.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.2.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv4_x.2.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.2.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 3
    scale = net.state_dict()['conv4_x.3.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.3.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv4_x.3.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.3.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 4
    scale = net.state_dict()['conv4_x.4.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.4.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv4_x.4.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.4.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 5
    scale = net.state_dict()['conv4_x.5.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.5.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv4_x.5.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.5.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # conv5_x
    # block 0 w/ short cut
    # residual_function
    current_c_temp = current_c
    scale = net.state_dict()['conv5_x.0.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.0.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv5_x.0.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.0.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    res_last_scale_map = scale > epsilon
    # shortcut
    scale = net.state_dict()['conv5_x.0.shortcut.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.0.shortcut.0.conv.weight'].numel() \
           * current_c_temp[0] * remaining_output_c // (current_c_temp[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    sc_scale_map = scale > epsilon
    remaining_output_c = (res_last_scale_map + sc_scale_map > 0).sum().tolist()
    # for module w/ shortcut, the logical and of both maps should be kept for next module
    current_c = (remaining_output_c, full_output_c)

    # block 1
    scale = net.state_dict()['conv5_x.1.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.1.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv5_x.1.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.1.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 2
    scale = net.state_dict()['conv5_x.2.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.2.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv5_x.2.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.2.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # fc
    full_para_num = net.state_dict()['fc.fc.weight'].numel() + net.state_dict()['fc.fc.bias'].numel()
    res += full_para_num * current_c[0] // current_c[1]

    return res


def para_counting_resnet34nbsp(net):
    res = 0
    current_c = (3, 3)

    # conv1
    scale = net.state_dict()['conv1.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv1.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # conv2_x
    # block 0
    scale = net.state_dict()['conv2_x.0.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.0.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv2_x.0.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.0.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 1
    scale = net.state_dict()['conv2_x.1.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.1.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv2_x.1.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.1.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 2
    scale = net.state_dict()['conv2_x.2.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.2.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv2_x.2.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv2_x.2.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # conv3_x
    # block 0 w/ shortcut
    # residual_function
    current_c_temp = current_c
    scale = net.state_dict()['conv3_x.0.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.0.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv3_x.0.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.0.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    res_last_scale_map = scale > epsilon
    # shortcut
    scale = net.state_dict()['conv3_x.0.shortcut.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.0.shortcut.0.conv.weight'].numel() \
           * current_c_temp[0] * remaining_output_c // (current_c_temp[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    sc_scale_map = scale > epsilon
    remaining_output_c = (res_last_scale_map + sc_scale_map > 0).sum().tolist()
    # for module w/ shortcut, the logical and of both maps should be kept for next module
    current_c = (remaining_output_c, full_output_c)

    # block 1
    scale = net.state_dict()['conv3_x.1.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.1.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv3_x.1.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.1.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 2
    scale = net.state_dict()['conv3_x.2.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.2.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv3_x.2.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.2.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 3
    scale = net.state_dict()['conv3_x.3.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.3.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv3_x.3.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv3_x.3.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # conv4_x
    # block 0 w/ short cut
    # residual_function
    current_c_temp = current_c
    scale = net.state_dict()['conv4_x.0.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.0.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv4_x.0.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.0.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    res_last_scale_map = scale > epsilon
    # shortcut
    scale = net.state_dict()['conv4_x.0.shortcut.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.0.shortcut.0.conv.weight'].numel() \
           * current_c_temp[0] * remaining_output_c // (current_c_temp[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    sc_scale_map = scale > epsilon
    remaining_output_c = (res_last_scale_map + sc_scale_map > 0).sum().tolist()
    # for module w/ shortcut, the logical and of both maps should be kept for next module
    current_c = (remaining_output_c, full_output_c)

    # block 1
    scale = net.state_dict()['conv4_x.1.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.1.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv4_x.1.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.1.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 2
    scale = net.state_dict()['conv4_x.2.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.2.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv4_x.2.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.2.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 3
    scale = net.state_dict()['conv4_x.3.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.3.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv4_x.3.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.3.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 4
    scale = net.state_dict()['conv4_x.4.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.4.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv4_x.4.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.4.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 5
    scale = net.state_dict()['conv4_x.5.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.5.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv4_x.5.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv4_x.5.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # conv5_x
    # block 0 w/ short cut
    # residual_function
    current_c_temp = current_c
    scale = net.state_dict()['conv5_x.0.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.0.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv5_x.0.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.0.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    res_last_scale_map = scale > epsilon
    # shortcut
    scale = net.state_dict()['conv5_x.0.shortcut.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.0.shortcut.0.conv.weight'].numel() \
           * current_c_temp[0] * remaining_output_c // (current_c_temp[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    sc_scale_map = scale > epsilon
    remaining_output_c = (res_last_scale_map + sc_scale_map > 0).sum().tolist()
    # for module w/ shortcut, the logical and of both maps should be kept for next module
    current_c = (remaining_output_c, full_output_c)

    # block 1
    scale = net.state_dict()['conv5_x.1.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.1.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv5_x.1.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.1.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # block 2
    scale = net.state_dict()['conv5_x.2.residual_function.0.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.2.residual_function.0.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    scale = net.state_dict()['conv5_x.2.residual_function.2.scale']
    full_output_c = scale.shape[0]
    remaining_output_c = (scale > epsilon).sum().tolist()
    res += net.state_dict()['conv5_x.2.residual_function.2.conv.weight'].numel() \
           * current_c[0] * remaining_output_c // (current_c[1] * full_output_c)
    # res += remaining_output_c * 2  # bn.weight + bn.bias
    current_c = (remaining_output_c, full_output_c)

    # fc
    full_para_num = net.state_dict()['fc.fc.weight'].numel() + net.state_dict()['fc.fc.bias'].numel()
    res += full_para_num * current_c[0] // current_c[1]

    return res


def para_counting_vgg19smbnsp(net):
    res = 0
    current_c = 3
    for m in net.modules():
        if isinstance(m, LinearScaled):  # final FC layer
            res += (current_c + 1) * m.fc.weight.shape[0]
            return res
        elif isinstance(m, Conv2dScaledWithBN):
            scale = m.scale
            remaining_output_c = (scale > epsilon).sum().tolist()
            kernel_size = m.conv.weight.shape[2] * m.conv.weight.shape[3]
            res += remaining_output_c * (current_c * kernel_size + 3)
            current_c = remaining_output_c


def para_counting(net, args):
    if args.net == 'resnet18nbsp':
        return para_counting_resnet18nbsp(net)
    if args.net == 'resnet18sp':
        return para_counting_resnet18sp(net)
    if args.net == 'vgg19smbnsp':
        return para_counting_vgg19smbnsp(net)
