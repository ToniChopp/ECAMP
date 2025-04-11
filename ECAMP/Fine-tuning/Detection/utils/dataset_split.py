import random
import os


def read_data_file(data_file):
    image_paths = []
    image_labels = []
    file_descriptor = open(data_file, "r")
    line = True
    while line:
        line = file_descriptor.readline()

        #--- if not empty
        if line:
            line_items = line.split()
            image_path = line_items[0]
            image_label = line_items[1:]
            image_label = [int(i) for i in image_label]
            image_paths.append(image_path)
            image_labels.append(image_label)
    file_descriptor.close()

    return image_paths, image_labels


def data_split(data_file, task, data_volume):
    image_paths, image_labels = read_data_file(data_file)
    train_len = len(image_paths)
    splits = []

    if data_volume == "1":            # 10 fold for 1% training data
        fold_size = round(train_len * 0.01)
        seeds = []
        for i in range(10):
            seeds.append(random.sample(range(0, 100), 1))

        for i in range(10):            # data split for 10 fold
            random.seed(seeds[i][0])
            random_split = random.sample(range(0, train_len), fold_size)
            splits.append(random_split)
        
        # save to file
        for i in range(10):
            file_name = "train_list_1_" + str(i) + ".txt"
            if not os.path.exists("./" + task + "/"):
                os.makedirs("./" + task + "/")
            file_path = os.path.join("./" + task + "/", file_name)
            file_descriptor = open(file_path, "w")
            for j in range(len(splits[i])):
                file_descriptor.write(image_paths[splits[i][j]] + " ")
                for k in range(len(image_labels[splits[i][j]])):
                    file_descriptor.write(str(image_labels[splits[i][j]][k]) + " ")
                file_descriptor.write("\n")
            file_descriptor.close()
    
    elif data_volume == "10":      # 5 fold for 10% training data
        fold_size = round(train_len * 0.1)
        seeds = []
        for i in range(5):
            seeds.append(random.sample(range(0, 100), 1))
        
        for i in range(5):            # data split for 5 fold
            random.seed(seeds[i][0])
            random_split = random.sample(range(0, train_len), fold_size)
            splits.append(random_split)

        # save to file
        for i in range(5):
            file_name = "train_list_10_" + str(i) + ".txt"
            if not os.path.exists("./" + task + "/"):
                os.makedirs("./" + task + "/")
            file_path = os.path.join("./" + task + "/", file_name)
            file_descriptor = open(file_path, "w")
            for j in range(len(splits[i])):
                file_descriptor.write(image_paths[splits[i][j]] + " ")
                for k in range(len(image_labels[splits[i][j]])):
                    file_descriptor.write(str(image_labels[splits[i][j]][k]) + " ")
                file_descriptor.write("\n")
            file_descriptor.close()
    
    else:
        return
        
