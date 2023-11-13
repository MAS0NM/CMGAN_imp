import torch
import openvino as ov
import numpy as np
import ipywidgets as widgets
import librosa
import soundfile as sf
import math
import os
from matplotlib import pyplot as plt
from openvino.runtime import Core
from models.dummy_generator import TSCNet as dummy
from utils import power_compress, power_uncompress


def load_ckpt_from_path(ckpt_path='./best_ckpt/ckpt'):
    net_dummy = dummy()
    net_dummy.load_state_dict(ckpt_path)
    return net_dummy


def wave2spec(wave):
    n_fft = 400
    hop = 100
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
    n_fft = 400
    hop = 100
    
    if not isinstance(est_real, torch.Tensor):
        est_real = torch.tensor(est_real)
    if not isinstance(est_imag, torch.Tensor):
        est_imag = torch.tensor(est_imag)
    est_real, est_imag = est_real.to(0), est_imag.to(0)
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
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    return est_audio


def inf(net, wav='../AudioSamples/noisy/p232_052.wav'):
    sample_audio, sr = librosa.load(wav)
    sample_spec, c, length = wave2spec(sample_audio)
    net = net.to(0)
    net.eval()
    est_real, est_imag = net(sample_spec)
    audio = post_process(est_real, est_imag, c, length)
    return audio, sr


def ckpt2onnx2ir(net, frame_size):
    onnx_path = f'./CMGAN_{frame_size}.onnx'
    ir_path = f'./CMGAN_{frame_size}.xml'
    
    if not os.path.exists(onnx_path): 
        dummy_input = torch.randn(1, 2, frame_size, 201)
        torch.onnx.export(net, dummy_input, onnx_path)
        
    if not os.path.exists(ir_path):
        ov_model = ov.convert_model(onnx_path)
        ov.save_model(ov_model, ir_path)
    
    
def spec_seg(spec, frame_size):
    l = spec.shape[2]
    n_segs = math.ceil(l / frame_size)
    print(n_segs)
    frame_list = []
    for idx in range(n_segs):
        if (idx+1)*frame_size <= l:
            frame = spec[:, :, idx*frame_size: (idx+1)*frame_size, :]
        else:
            frame = spec[:, :, idx*frame_size: , :]
            npad = ((0,0), (0,0), (0,frame_size-l%frame_size), (0,0))
            frame = np.pad(frame, pad_width=npad, mode='constant', constant_values=0)
        frame_list.append(frame)
    print('np.array(frame_list).shape', np.array(frame_list).shape)
    return frame_list
    

def inf(spec, compiled_model, c, length):
    res = compiled_model([ov.Tensor(spec)])
    est_real, est_imag = torch.from_numpy(res[compiled_model.output(0)]), torch.from_numpy(res[compiled_model.output(1)])
    return est_real, est_imag


def do_inf(spec, compiled_model, frame_size, c, length):
    frame_list = spec_seg(spec, frame_size)
    reals, imags = [], []
    for frame in frame_list:
        real, imag = inf(frame, compiled_model, c, length)
        reals.append(real)
        imags.append(imag)
    est_real = np.concatenate(reals, axis=2)
    print(est_real.shape)
    est_imag = np.concatenate(imags, axis=2)
    print(est_imag.shape)
    audio = post_process(torch.tensor(est_real).to(0), torch.tensor(est_imag).to(0), c, length)
    sf.write(f'./exp_{frame_size}.wav', audio, samplerate=16000)
    

if __name__ == '__main__':
    
    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
        
    for frame_size in [1, 10, 50, 100, 200, 300]:
        net = dummy()
        ckpt2onnx2ir(net, frame_size)
        model = core.read_model(
            model=f'CMGAN_{frame_size}.xml'
        )
        compiled_model = core.compile_model(model=model, device_name=device.value)
        sample_audio, sr = librosa.load('../AudioSamples/noisy/p232_052.wav', sr=16000)
        # sample_audio = sample_audio[:len(sample_audio)//2]
        sample_spec, c, length = wave2spec(sample_audio)
        print(sample_spec.shape)
        # convert to tensor here becuz ov.Tensor() accepts np not torch.Tensor
        sample_spec = sample_spec.cpu().numpy()
        
        audio = do_inf(sample_spec, compiled_model, frame_size, c, length)