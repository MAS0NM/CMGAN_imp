import openvino as ov
import torch
import soundfile as sf
import librosa
import numpy as np
import ipywidgets as widgets
import os

from utils import power_compress, power_uncompress
from models.generator import TSCNet, DilatedDenseNet
from models.dummy_generator import TSCNet as dummy

# net_G = TSCNet()
net_dummy = dummy()


weight_pth = './best_ckpt/ckpt'
weight = torch.load(weight_pth)
# net_G.load_state_dict(weight)
net_dummy.load_state_dict(weight)

n_fft = 400
hop = 100


def wave2spec(wave):
    noisy = torch.tensor(wave).unsqueeze(0).to(0)
    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    length = noisy.size(-1)
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)
    noisy_spec = torch.stft(
        noisy,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).to(0),
        onesided=True,
        return_complex=False
    )
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    return noisy_spec, c, length


@torch.no_grad()
def post_process(est_real, est_imag, c, length):
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    est_audio = torch.istft(
        est_spec_uncompress,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).to(0),
        onesided=True,
        return_complex=False
    )
    # est_audio = np.array(est_audio.squeeze(0).cpu().detach())
    
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    return est_audio


def get_all_layers(net):
    layers = []
    for mod_name, module in net.named_modules():
        for lyr_name, layer in module.named_modules():
            # if hasattr(layer, 'weight') and 'prelu' not in lyr_name:
            if hasattr(layer, 'weight'):
                weights = layer.weight.data
                layers.append([lyr_name, weights])
    return layers


def weight_transfer(net_src, net_tar):
    net_G_layers = get_all_layers(net_src)
    net_dummy_layers = get_all_layers(net_tar)
    print(f'original layers: {len(net_G_layers)}, dummy layers: {len(net_dummy_layers)}')
    num_layer_transfered = 0
    for dum_mod_name, module in net_tar.named_modules():
        for dum_lyr_name, layer in module.named_modules():
            if hasattr(layer, 'weight'):
                name = net_G_layers[0][0]
                name, weights = net_G_layers.pop(0)
                num_layer_transfered += 1
                layer.weight.data = weights
            
    print('weights transfer done')
    print(f'num of modules transfered: {num_layer_transfered}')
    
    
def inf(net, wav='../AudioSamples/noisy/p232_052.wav'):
    sample_audio, sr = librosa.load(wav)
    sample_spec, c, length = wave2spec(sample_audio)
    net = net.to(0)
    net.eval()
    est_real, est_imag = net(sample_spec)
    audio = post_process(est_real, est_imag, c, length)
    return audio, sr
    
    
'''
    dummy input test
'''
# input_test = torch.randn(1, 2, 99, 201)
# a, b = net_dummy(input_test)
# print(a.shape, b.shape)
# input_test = torch.randn(1, 2, 99, 201)
# res = net_G(input_test)


'''
    onnx export test since direct IR conversion has issue
'''
frame_size = 3200
onnx_path = f'CMGAN_{frame_size}.onnx'
ir_path = f'CMGAN_{frame_size}.xml'
if not os.path.exists(onnx_path):
    sample_audio, sr = librosa.load('../AudioSamples/noisy/p232_052.wav', sr=16000)
    sample_audio = sample_audio[:frame_size]
    sample_spec, c, length = wave2spec(sample_audio)
    # sample_spec = torch.randn(1, 2, 485, 201).to(0)
    sample_spec = sample_spec.cpu()
    print(sample_spec.shape)
    torch.onnx.export(net_dummy, sample_spec, onnx_path)

if not os.path.exists(ir_path):
    ov_model = ov.convert_model(onnx_path)
    ov.save_model(ov_model, ir_path)


'''
    direct IR converison
'''
# core = ov.Core()
# ov_model = ov.convert_model(net_dummy)
# ov.save_model(ov_model, './IR_model')


# device = widgets.Dropdown(
#     options=core.available_devices + ["AUTO"],
#     value='AUTO',
#     description='Device:',
#     disabled=False,
# )
# compiled_model = core.compile_model(ov_model, device.value)



'''
    load the ckpt to original network 
    and transfer the weights to dummy network 
    for IR conversion
'''

# weight_transfer(net_G, net_dummy)
# for name, module in net_dummy.named_children():
#     if isinstance(module, torch.nn.PReLU):
#         setattr(model, name, torch.nn.LeakyReLU())
#     else:
#         replace_prelu_with_leakyrelu(module)
        
# PReLUlist = []
# for mod_name, module in net_dummy.named_modules():
#     for lyr_name, layer in module.named_modules():
#         if isinstance(layer, torch.nn.PReLU):
#             if mod_name and lyr_name:
#                 setattr(net_dummy, lyr_name, torch.nn.LeakyReLU())
# print(PReLUlist)
# print(len(get_all_layers(net_dummy)))
# for name in PReLUlist:
#     # setattr(net_dummy, name, torch.nn.LeakyReLU())
#     name_parts = name.strip().split('.')
#     sub_module = net_dummy
#     for part in name_parts:
#         sub_module = getattr(sub_module, part)
#     setattr(sub_module, name_parts[-1], torch.nn.LeakyReLU())
    
# print(len(get_all_layers(net_dummy)))


# PReLUlist = []
# for mod_name, module in net_dummy.named_modules():
#     for lyr_name, layer in module.named_modules():
#         if isinstance(layer, torch.nn.PReLU):
#             layer_name = f'{mod_name}.{lyr_name}' if lyr_name else mod_name
#             PReLUlist.append(layer_name)
# print(PReLUlist)