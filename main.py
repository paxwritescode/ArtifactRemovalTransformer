import utils
import os

if __name__ == '__main__':
    # parameter setting
    input_path = './sampledata/'
    input_name = 'sampledata.csv'
    sample_rate = 256 # input data sample rate
    modelname = 'ICUNet_attn' # or 'ICUNet', 'ICUNet++', 'ICUNet_attn', 'ART'
    output_path = './sampledata/'
    output_name = 'outputsample.csv'


    # step1: Data preprocessing
    preprocess_data = utils.preprocessing(input_path+input_name, sample_rate)

    # step2: Signal reconstruction
    utils.reconstruct(modelname, preprocess_data, output_path+output_name)
    