import os
import pickle

import pinyin
from tqdm import tqdm

from config import data_file, wav_folder, tran_file


# split in ['train', 'test', 'dev']
def get_aishell_data(split):
    print('loading {} samples...'.format(split))

    with open(tran_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    tran_dict = dict()
    for line in lines:
        tokens = line.split()
        key = tokens[0]
        trn = ''.join(tokens[1:])
        tran_dict[key] = trn

    samples = []

    folder = os.path.join(wav_folder, split)
    dirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    for dir in tqdm(dirs):
        files = [f for f in os.listdir(dir) if f.endswith('.wav')]

        for f in files:
            audiopath = os.path.join(dir, f)

            key = f.split('.')[0]
            if key in tran_dict:
                text = tran_dict[key]
                text = pinyin.get(text, format="numerical", delimiter=" ")

                samples.append({'audiopath': audiopath, 'text': text})

    return samples


def main():
    data = dict()
    data['train'] = get_aishell_data('train')
    data['dev'] = get_aishell_data('dev')
    data['test'] = get_aishell_data('test')

    with open(data_file, 'wb') as file:
        pickle.dump(data, file)

    print('num_train: ' + str(len(data['train'])))
    print('num_dev: ' + str(len(data['dev'])))
    print('num_test: ' + str(len(data['test'])))


if __name__ == "__main__":
    main()
