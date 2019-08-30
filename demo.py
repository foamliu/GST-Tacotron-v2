import matplotlib.pylab as plt
import numpy as np
import pinyin
import soundfile as sf
import torch

from config import sampling_rate
from utils import text_to_sequence, ensure_folder, plot_data, Denoiser

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model.eval()

    waveglow_path = 'waveglow_256channels.pt'
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)

    text = "必须树立公共交通优先发展的理念"
    text = pinyin.get(text, format="numerical", delimiter=" ")
    print(text)
    sequence = np.array(text_to_sequence(text))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
               mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T))

    ensure_folder('images')
    plt.savefig('images/mel_spec.jpg')

    mel_outputs_postnet = mel_outputs_postnet.type(torch.float16)
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

    audio = audio[0].data.cpu().numpy()
    audio = audio.astype(np.float32)

    print('audio.shape: ' + str(audio.shape))
    print(audio)

    sf.write('output.wav', audio, sampling_rate, 'PCM_24')
