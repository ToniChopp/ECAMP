import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import ast
from sklearn.model_selection import train_test_split

# label_file = "../../../../Data/RSNA/stage_2_train_labels.csv"

# def create_bbox(row):
#     if row["Target"] == 0:
#         return 0
#     else:
#         x1 = row["x"]
#         y1 = row["y"]
#         x2 = x1 + row["width"]
#         y2 = y1 + row["height"]
#         return [x1, y1, x2, y2]
    
# def modi_path(id):
#     path = "train_images/" + id + ".png"
#     return path


# df = pd.read_csv(label_file)
# df["bbox"] = df.apply(lambda x: create_bbox(x), axis=1)
# df["image_path"] = df["patientId"].apply(lambda x: modi_path(x))
# df = df[["image_path", "bbox"]]
# df = df.groupby("image_path").agg(list)
# print(len(df))
# df = df.reset_index()
# df["bbox"] = df["bbox"].apply(lambda x: [[0.0,0.0,0.,0.]] if x == [0] else x)

# train_df, test_val_df = train_test_split(df, test_size=5337 * 2, random_state=0)
# test_df, val_df = train_test_split(test_val_df, test_size=0.5, random_state=0)

# train_df.to_csv("./train_list.csv", index=False)
# test_df.to_csv("./test_list.csv", index=False)
# val_df.to_csv("./val_list.csv", index=False)


train_file = "./train_list_1.csv"
df = pd.read_csv(train_file)
bboxes = df["bbox"]
import ipdb; ipdb.set_trace()
for bbox in bboxes:
    bbox = ast.literal_eval(bbox)
    bbox = np.array(bbox)
    new_bbox = bbox[bbox[:, 3] > 0].astype(np.int64)