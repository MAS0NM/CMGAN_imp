import openvino as ov
import torch
import soundfile as sf
import librosa
import ipywidgets as widgets
import numpy as np
from utils import power_compress, power_uncompress


'''
    onnx runtime
'''

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
    
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    return est_audio


def run(spec, compiled_model):
    res = compiled_model([ov.Tensor(spec)])
    est_real, est_imag = torch.from_numpy(res[compiled_model.output(0)]), torch.from_numpy(res[compiled_model.output(1)])
    audio = post_process(torch.tensor(est_real).to(0), torch.tensor(est_imag).to(0), c, length)
    return audio


core = ov.Core()
device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)
model = core.read_model(
    model='CMGAN_wavsize_32000.xml'
)

compiled_model = core.compile_model(model=model, device_name=device.value)

sample_audio, sr = librosa.load('../AudioSamples/noisy/p232_052.wav', sr=16000)
sample_audio = sample_audio[:32000]
sample_spec, c, length = wave2spec(sample_audio)
sample_spec = sample_spec.cpu().numpy()

audio = run(sample_spec, compiled_model)

print('writing file into test_IR.wav')
sf.write('test_IR.wav', audio, samplerate=sr)