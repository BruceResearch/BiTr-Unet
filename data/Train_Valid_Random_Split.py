import os
import shutil
import numpy as np

source1 = "/scratch/qj2022/TransBTS-main-2/data/BraTS2021_TrainingData"
dest11 = "/scratch/qj2022/TransBTS-main-2/data/BraTS2021_ValidationData"
files = os.listdir(source1)

for f in files:
    if np.random.rand(1) < 0.2:
        shutil.move(source1 + '/'+ f, dest11 + '/'+ f)