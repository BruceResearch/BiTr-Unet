import os
import numpy as np


subjects = [ f.name for f in os.scandir('/path to main directory/data/BraTS2021_ValidationData') if f.is_dir() ]
with open('official_valid.txt', 'w') as f:
    for item in subjects:
        f.write("%s\n" % item)
