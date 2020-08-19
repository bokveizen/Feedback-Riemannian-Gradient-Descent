def cmd_maker(gpu_id, basenet, net_suffix, additional_para):
    # CUDA_VISIBLE_DEVICES=0 python train.py -n resnet18nbsp -sf -t 0.5
    return 'CUDA_VISIBLE_DEVICES={} python train.py -n {}{} {}'.format(gpu_id, basenet, net_suffix, additional_para)


gpu_list = [2, 3, 4, 5]
gpu_num = len(gpu_list)
cmd_list = [''] * gpu_num
para_list = [
    ('', ''),
    ('', '-of'),
    ('', '-sf'),
    ('wn', ''),
    ('wnc', ''),
    ('wncw', ''),
    ('wnccw', ''),
    ('s', '-of'),
    ('s', '-sf'),
]
threshold_rate_list = [0.1, 0.15,
                       0.2, 0.25,
                       0.3, 0.325, 0.35, 0.375,
                       0.4, 0.425, 0.45, 0.475,
                       0.5, 0.525, 0.55, 0.575,
                       0.6, 0.625, 0.65, 0.675,
                       0.7, 0.75,
                       0.8, 0.85,
                       0.9]
for tr in threshold_rate_list:
    para_list.append(('sp', '-of -t {}'.format(tr)))
    para_list.append(('sp', '-sf -t {}'.format(tr)))

basenet = 'resnet18'
current_gpu_index = 0
for para in para_list:
    gpu_id = gpu_list[current_gpu_index]
    net_suffix = para[0]
    additional_para = para[1]
    if '-sf' in additional_para:
        additional_para += ' -stp 5e-4'
        cmd = cmd_maker(gpu_id, basenet, net_suffix, additional_para)
        cmd_list[current_gpu_index] += cmd + '; '
        current_gpu_index = (current_gpu_index + 1) % gpu_num
for cmd in cmd_list:
    print(cmd)


