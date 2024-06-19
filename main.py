import utils
import os

if __name__ == '__main__':
    # parameter setting
    sub = 2
    input_path = 'G:/共用雲端硬碟/CNElab_專題111_ArtifactRemoval/6.Opendataset/2.EEG_Motor_Movement_Imagery/2.rawdata/s_csv/S{:03d}/'.format(sub)
    #input_path = 'G:/共用雲端硬碟/CNElab_專題111_ArtifactRemoval/6.Opendataset/4.BETA_SSVEP/2.rawdata/r_csv/'
    sample_rate = 256  # input data sample rate
    modelname = ['ICUNet', 'UNetpp', 'AttUnet', 'EEGART', 'ResCNN', 'DuoCL', 'GCTNet']
    #idx = [3, 4, 7, 8, 11, 12]

    for i in range(6):
        i=i+0
        output_path = 'G:/共用雲端硬碟/CNElab_專題111_ArtifactRemoval/6.Opendataset/2.EEG_Motor_Movement_Imagery/2.rawdata/s_csv/S{:03d}/'.format(sub)+modelname[i]+'_csv/'
        #output_path = 'G:/共用雲端硬碟/CNElab_專題111_ArtifactRemoval/6.Opendataset/4.BETA_SSVEP/3.decodecsv/'+modelname[i]+'_csv/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for j in range(22):
            j=j+0
            input_name = 'mapped_{:02d}.csv'.format(j+1)
            output_name = '{}_{:02d}.csv'.format(modelname[i], j+1)

            # step1: Data preprocessing
            total_file_num = utils.preprocessing(input_path+input_name, sample_rate)

            # step2: Signal reconstruction
            utils.reconstruct(modelname[i], total_file_num, output_path+output_name)
