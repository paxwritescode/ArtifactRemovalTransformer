# ArtifactRemovalTransformer (ART)

# Quickstart

## 1. Channel mapping

### Raw data
1. The data need to be a two-dimensional array (channel, timepoint).
2. Make sure you have **resampled** your data to **256 Hz**.
3. Upload your EEG data in `.csv` format.

### Channel locations
Upload your data's channel locations in `.loc` format, which can be obtained using **EEGLAB**.  
>If you cannot obtain it, we recommend you to download the standard montage <a href="">here</a>. If the channels in those files doesn't match yours, you can use **EEGLAB** to modify them to your needed montage.

### Imputation
The models was trained using the EEG signals of 30 channels, including: `Fp1, Fp2, F7, F3, Fz, F4, F8, FT7, FC3, FCz, FC4, FT8, T7, C3, Cz, C4, T8, TP7, CP3, CPz, CP4, TP8, P7, P3, Pz, P4, P8, O1, Oz, O2`.
We expect your input data to include these channels as well.  
If your data doesn't contain all of the mentioned channels, there are 3 imputation ways you can choose from:

<u>Manually</u>:  
- **mean**: select the channels you wish to use for imputing the required one, and we will average their values. If you select nothing, zeros will be imputed. For example, you didn't have **FCZ** and you choose **FC1, FC2, FZ, CZ** to impute it(depending on the channels you have), we will compute the mean of these 4 channels and assign this new value to **FCZ**.

<u>Automatically</u>:  
Firstly, we will attempt to find neighboring channel to use as alternative. For instance, if the required channel is **FC3** but you only have **FC1**, we will use it as a replacement for **FC3**.  
Then, depending on the **Imputation** way you chose, we will:
- **zero**: fill the missing channels with zeros.
- **adjacent**: fill the missing channels using neighboring channels which are located closer to the center. For example, if the required channel is **FC3** but you only have **F3, C3**, then we will choose **C3** as the imputing value for **FC3**.
>Note: The imputed channels **need to be removed** after the data being reconstructed.

### Mapping result
Once the mapping process is finished, the **template montage** and the **input montage**(with the channels choosen by the mapping function displaying their names) will be shown.

### Missing channels
The channels displayed here are those for which the template didn't find suitable channels to use, and utilized **Imputation** to fill the missing values.  
Therefore, you need to
<span style="color:red">**remove these channels**</span>
after you download the denoised data.

### Template location file
You need to use this as the **new location file** for the denoised data.

## 2. Decode data

### Model
Select the model you want to use.  
The detailed description of the models can be found in other pages.
