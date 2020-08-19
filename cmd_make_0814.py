def cmd_maker(gpu_id, basenet, net_suffix, additional_para):
    # CUDA_VISIBLE_DEVICES=0 python train.py -n resnet18nbsp -sf -t 0.5
    return 'CUDA_VISIBLE_DEVICES={} python train.py -n {}{} {}; '.format(gpu_id, basenet, net_suffix, additional_para)


gpu_list = [0, 1, 2]
# gpu_list = [0, 1]
gpu_num = len(gpu_list)
cmd_list = [''] * gpu_num
para_list = [
    ('', ''),
    ('', '-of'),
    ('', '-sf'),
    ('wn', ''),
    ('wncw', ''),
    ('s', '-of'),
    ('s', '-sf'),
    # ('wn2', ''),
    # ('wn2', '-of'),
    # ('wn2', '-sf'),
    # ('wn2s', '-of'),
    # ('wn2s', '-sf'),
]

# threshold_rate_list = [
#     0.1,
#     0.2,
#     0.3,
#     0.4,
#     0.5, 0.525, 0.55, 0.575,
#     0.6, 0.625, 0.65, 0.675,
#     0.7, 0.725, 0.75, 0.775,
#     0.8,
#     0.9
# ]
# for tr in threshold_rate_list:
#     # para_list.append(('sp', '-of -t {}'.format(tr)))
#     # para_list.append(('sp', '-sf -t {}'.format(tr)))
#     para_list.append(('wn2sp', '-of -t {}'.format(tr)))
#     para_list.append(('wn2sp', '-sf -t {}'.format(tr)))

basenet_list = [
    'vgg11sm',
    'vgg11smbn',
    'vgg13sm',
    'vgg13smbn',
    # 'vgg16sm',
    'vgg16smbn',
    # 'vgg19sm',
    'vgg19smbn',
    'resnet18',
    'resnet18nb',
    'resnet34',
    'resnet34nb',
    'resnet50',
    'resnet50nb',
]

current_gpu_index = 0
for basenet in basenet_list:
    for para in para_list:
        gpu_id = gpu_list[current_gpu_index]
        net_suffix = para[0]
        additional_para = para[1] + ' -m 0.0 -l 0.4'
        cmd = cmd_maker(gpu_id, basenet, net_suffix, additional_para)
        cmd_list[current_gpu_index] += cmd
        current_gpu_index = (current_gpu_index + 1) % gpu_num
for cmd in cmd_list:
    print(cmd)
