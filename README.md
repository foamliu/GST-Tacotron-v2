# Tacotron 2

A PyTorch implementation of Tacotron2, described in [Natural TTS Synthesis By Conditioning Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf), an end-to-end text-to-speech(TTS) neural network architecture, which directly converts character text sequence to speech.

## Dataset

[Aishell Dataset](http://www.openslr.org/33/), containing 400 speakers and over 170 hours of Mandarin speech data.

## Dependency

- Python 3.5.2
- PyTorch 1.0.0

## Usage
### Data Pre-processing
Extract data_aishell.tgz:
```bash
$ python extract.py
```

Extract wav files into train/dev/test folders:
```bash
$ cd data/data_aishell/wav
$ find . -name '*.tar.gz' -execdir tar -xzvf '{}' \;
```

Scan transcript data, generate features:
```bash
$ python pre_process.py
```

Now the folder structure under data folder is sth. like:
<pre>
data/
    data_aishell.tgz
    data_aishell/
        transcript/
            aishell_transcript_v0.8.txt
        wav/
            train/
            dev/
            test/
    aishell.pickle
</pre>

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

### Demo
Generate mel-spectrogram for text "Waveglow is really awesome!"
```bash
$ python demo.py
```
![image](https://github.com/foamliu/Tacotron2-CN/raw/master/images/mel_spec.jpg)
