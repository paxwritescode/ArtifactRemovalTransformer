import csv
import numpy as np


def read_train_data(file_name):
    with open(file_name, 'r', newline='') as f:
        lines = csv.reader(f)
        data = []
        for line in lines:
            data.append(line)

    data = np.array(data).astype(np.float64)
    return data


def save_data(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


# create your folder
input_path = './sampledata/'
output_path = './sampledata/'
# name your file
input_name = 'sampledata.csv' # Here is a sample for the batch process: 'raw_{:03d}.csv'.format(i+1)
output_name = 'outputsample.csv'# Here is a sample for the batch process:'mapped_{:03d}.csv'.format(i+1)

# read input data
old_data = read_train_data(input_path+input_name)
# create output zeros matrix
new_data = np.zeros((30, old_data.shape[1]))

# ** key step ** Channel Mapping
old_idx = [ 22,  24, 30, 32, 34, 36, 38,  39,   2,   4,   6,  40, 41,  9, 11, 13, 42,  45,  16,  18,  20,  46, 47, 49, 51, 53, 55, 61, 62, 63]

# channel mapping
for j in range(30):
    new_data[j, :] = old_data[old_idx[j]-1, :]

# save the data
save_data(new_data, output_path+output_name)
