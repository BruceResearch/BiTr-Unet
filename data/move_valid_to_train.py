import os
import shutil
import numpy as np

source1 = "./data/BraTS2021_ValidationData"
dest11 = "./data/BraTS2021_TrainingData"
files = os.listdir(source1)

for f in files:
    
    shutil.move(source1 + '/'+ f, dest11 + '/'+ f)