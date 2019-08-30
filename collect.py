import os
from shutil import copyfile

from config import thchs30_folder
from utils import ensure_folder

folder = os.path.join(thchs30_folder, 'train')
files = [f for f in os.listdir(folder) if f.endswith('.wav')]
filename = os.path.join(folder, files[0])

ensure_folder('audios')
target = os.path.join('audios', files[0])
copyfile(filename, target)
