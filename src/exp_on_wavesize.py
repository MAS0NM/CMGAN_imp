import torch
import openvino as ov
import numpy as np
import ipywidgets as widgets
import librosa
import soundfile as sf
import math
import os
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


def wave_seg(wave, frame_size):
    L = len(wave)
    n_segs = math.ceil(L / frame_size)
    frame_list = []
    bias = frame_size * 0.3 // 1
    bias = 0
    for idx in range(n_segs):
        if idx == 0:
            frame = wave[idx*frame_size: (idx+1)*frame_size]
        elif (idx+1) * frame_size <= L and idx != 0:
            frame = wave[idx*frame_size-bias: (idx+1)*frame_size]
        else:
            frame = wave[idx*frame_size-bias: ]
            npad = ((0, frame_size-L%frame_size))
            frame = np.pad(frame, pad_width=npad, mode='constant', constant_values=0)
        frame_list.append(frame)
    return frame_list
    

@torch.no_grad()
def post_process(est_real, est_imag, c, length):
    n_fft = 400
    hop = 100
    
    if not isinstance(est_real, torch.Tensor):
        est_real = torch.tensor(est_real).to(0)
    if not isinstance(est_imag, torch.Tensor):
        est_imag = torch.tensor(est_imag).to(0)
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


def ckpt2onnx2ir(net, frame_size, model_name):
    net = net.to(0)
    onnx_path = f'{model_name}.onnx'
    ir_path = f'{model_name}.xml'
    sample_audio, sr = librosa.load('../AudioSamples/noisy/p232_052.wav', sr=16000)
    sample_audio = sample_audio[:frame_size]
    # print(f'dummpy wave shape: {sample_audio.shape}')
    spec, c, length = wave2spec(sample_audio)
    spec = torch.tensor(spec).to(0)
    
    # print(f'spec shape: {spec.shape}')
    if not os.path.exists(onnx_path): 
        # dummy_input = torch.randn(1, 2, spec.shape[2], 201)
        torch.onnx.export(net, spec, onnx_path)
        
    if not os.path.exists(ir_path):
        ov_model = ov.convert_model(onnx_path)
        ov.save_model(ov_model, ir_path)    


def inf(spec, compiled_model):
    res = compiled_model([ov.Tensor(spec)])
    est_real, est_imag = torch.from_numpy(res[compiled_model.output(0)]), torch.from_numpy(res[compiled_model.output(1)])
    return est_real, est_imag


def do_inf(wave, compiled_model, frame_size):
    audio_segs = []
    
    L = len(wave)
    
    
    # wave = wave[:frame_size]
    # spec, c, length = wave2spec(wave)
    # spec = spec.cpu().numpy()
    # real, imag = inf(spec, compiled_model)
    # audio = post_process(torch.tensor(real).to(0), torch.tensor(imag).to(0), c, length)
    #     
    
    
    
    frame_list = wave_seg(wave, frame_size)
    for frame in frame_list:
        spec, c, length = wave2spec(frame)
        spec = spec.cpu().numpy()
        real, imag = inf(spec, compiled_model)
        audio = post_process(torch.tensor(real).to(0), torch.tensor(imag).to(0), c, length)
        audio_segs.append(audio)
    audio = np.concatenate(audio_segs)[:L]
    
    print(f'output audio shape: {audio.shape}')
    sf.write(f'./exp_{frame_size}.wav', audio, samplerate=16000)

    
if __name__ == '__main__':
    
    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    for wav_size in [3200]:
        # model_name = f'CMGAN_wavsize_{wav_size}'
        model_name = f'CMGAN_{wav_size}'
        net = dummy()
        # ckpt2onnx2ir(net, wav_size, model_name)
        model = core.read_model(
            model=f'{model_name}.xml'
        )
        compiled_model = core.compile_model(model=model, device_name=device.value)
        sample_audio, sr = librosa.load('../AudioSamples/noisy/p257_011.wav', sr=16000)
        print(f'origin wav shape: {sample_audio.shape}')
        # convert to tensor here becuz ov.Tensor() accepts np not torch.Tensor
        # sample_audio = sample_audio.cpu().numpy()
        
        audio = do_inf(sample_audio, compiled_model, wav_size)