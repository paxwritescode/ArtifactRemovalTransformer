import utils
import os

if __name__ == '__main__':
    # parameter setting
    input_path = './data/'
    input_name = 'eeg_test_converted.csv'
    sample_rate = 500 # input data sample rate
    modelname = 'ART' # or 'ICUNet', 'ICUNet++', 'ICUNet_attn', 'ART'
    output_path = './data/'
    output_name = 'our_output.csv'

    # read the mapping result
    mapping_name = './data/mapped_output.json'
    mapping_result, num_channel, num_group = utils.read_mapping_result(mapping_name)

    for i in range(num_group):

        # step1: Data preprocessing
        preprocess_data = utils.preprocessing(input_path+input_name, sample_rate, mapping_result[i])
        # step2: Signal reconstruction
        reconstructed_data = utils.reconstruct(modelname, preprocess_data, output_name, i)
        # step3: Data postprocessing
        utils.postprocessing(reconstructed_data, sample_rate, output_path+output_name, mapping_result[i], i, num_channel)
