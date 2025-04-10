import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import ast
from sklearn.model_selection import train_test_split
import ipdb


def rle2mask(rle, width, height):
    """Run length encoding to segmentation mask"""

    ipdb.set_trace()
    mask = np.zeros(width * height)
    array = np.array([int(x) for x in rle.split(" ")])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position: current_position + lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height).T

train_file = "./val_list.csv"
df = pd.read_csv(train_file)
rle_list = df["EncodedPixels"]

ipdb.set_trace()
img_ids = df["ImageId"].unique().tolist()

img_id = img_ids[1]

img_id_df = df.groupby("ImageId").get_group(img_id)

rle_list = img_id_df["EncodedPixels"].tolist()
mask = np.zeros([1024, 1024])
if rle_list[0] != "-1":
    for rle in rle_list:
        mask += rle2mask(rle, 1024, 1024)
