import utils
import os

from dataclasses import dataclass

@dataclass
class InputConfig:
    input_path: str
    input_name: str
    sample_rate: int
    output_path: str
    output_name: str
    mapping_result: str


if __name__ == '__main__':
    # parameter setting
    our_input = False
    ic: InputConfig
    if our_input:
        ic = InputConfig(
            input_path = './data/',
            input_name = 'eeg_test_converted.csv',
            sample_rate = 500, # input data sample rate
            output_path = './data/',
            output_name = 'our_output.csv',
            mapping_result = './data/mapped_output.json'
        )
    else:
        ic = InputConfig(
            input_path = './sampledata/',
            input_name = 'sampledata.csv',
            sample_rate = 256,
            output_path = './sampledata/',
            output_name ='outputsample.csv',
            mapping_result = './sampledata/sample_chanlocs_mapping_result.json'
        )
    modelname = 'ART'
        

    # read the mapping result
    mapping_result, num_channel, num_group = utils.read_mapping_result(ic.mapping_result)

    for i in range(num_group):

        # step1: Data preprocessing
        preprocess_data = utils.preprocessing(ic.input_path+ic.input_name, ic.sample_rate, mapping_result[i])
        # step2: Signal reconstruction
        reconstructed_data = utils.reconstruct(modelname, preprocess_data, ic.output_name, i)
        # step3: Data postprocessing
        utils.postprocessing(reconstructed_data, ic.sample_rate, ic.output_path+ic.output_name, mapping_result[i], i, num_channel)
