#!/usr/bin/env python
# coding: utf-8

# In[115]:


import pandas as pd
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[3]:


def downsampling_df(df,n,labels,target_column,random_state=1):
    """
        Downsample a Dataframe to ensure that all classes represented by the 'labels'
        have at most 'n' samples. If a class has fewer than 'n' samples, it includes all of them.
    
        Parameters:
        df (pandas.DataFrame): The dataframe to downsample.
        n (int): The maximum number of samples for each label after downsampling.
        labels (list): A list of unique labels/classes to downsample in the dataframe.
        target_column (str): The name of the column in 'df' that contains the class labels.
        random_state (int, optional): The seed used by the random number generator for reproducibility.
    
        Returns:
        pandas.DataFrame: A new dataframe where each class has been downsampled to at most 'n' samples.
    
        Raises:
        ValueError: If 'df' is not a pandas DataFrame, 'n' is not an integer, 'labels' is not a list,
                    'target_column' is not a string, or 'target_column' does not exist in 'df'.
    
        Example:
        >>> df = pd.DataFrame({'class': ['A', 'A', 'B', 'B', 'C'], 'data': [1, 2, 3, 4, 5]})
        >>> downsampling_df(df, n=1, labels=['A', 'B', 'C'], target_column='class', random_state=42)
           class  data
        1     A     2
        3     B     4
        4     C     5
    
        Note:
        The function allows setting a 'random_state' for sampling to ensure reproducibility. If 'random_state' is
        not provided, it defaults to 1.
    """
    
    downsampled_dfs = []
    for i in labels:
        i_df = df[df[target_column] == i]
        if len(i_df) >= n:
            downsampled_df = i_df.sample(n=n, random_state=random_state)
        else:
            downsampled_df = i_df
        downsampled_dfs.append(downsampled_df)
    result_df = pd.concat(downsampled_dfs)
    return result_df


# In[14]:


class SoundProcessor():
    """
        A class to process audio data for machine learning purposes.
    
        This class provides functionality to process audio files stored in a specified
        directory and convert them into spectrograms or Mel-frequency cepstral coefficients (MFCCs).
        The processed data can then be used for audio classification tasks with machine learning models.
    
        Attributes:
        df (pandas.DataFrame): A DataFrame containing metadata for the audio files,
                               including the file indices and target labels.
        path (str): The directory path where the audio files are stored.
        target (str): The column name in 'df' that contains the target labels for classification.
        n_fft (int): The number of data points used in each block for the FFT (default is 2048).
        hop_length (int): The number of samples between successive frames (default is 512).
    
        Methods:
        process_and_save_spectrograms(path, batch_size=10):
            Processes audio files in batches, computes their short-time Fourier transform (STFT)
            spectrograms, and saves the spectrograms to disk. The spectrograms are saved in directories
            named after their corresponding labels.
    
        mfcc(batch_size=10):
            Processes audio files in batches and computes the MFCCs, which are returned as a
            pandas DataFrame. This DataFrame can be used as feature data for machine learning models.
    
        Example:
        >>> df = pd.DataFrame({'file_index': ['001', '002'], 'target_label': ['piano', 'violin']})
        >>> processor = SoundProcessor(df, 'path/to/audio', 'target_label')
        >>> processor.process_and_save_spectrograms('path/to/spectrograms')
        >>> mfcc_features = processor.mfcc()
    """
    
    def __init__(self, df, path, target, n_fft=2048, hop_length=512):
        self.df = df
        self.path = path
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target = target
    
    def process_and_save_spectrograms(self, path, batch_size=10):
        # Create the output directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Process in batches
        for start_idx in range(0, len(self.df), batch_size):
            end_idx = min(start_idx + batch_size, len(self.df))
            for local_idx, global_idx in enumerate(range(start_idx, end_idx), start_idx):
                # Load the audio file
                audio_data, sampling_rate = librosa.load(
                    os.path.join(self.path, f'{self.df.index[global_idx]}.wav'),
                    sr=None
                )
                
                # Compute the STFT
                fourier_t = librosa.core.stft(audio_data, hop_length=self.hop_length, n_fft=self.n_fft)
                spectrogram = np.abs(fourier_t)
                log_spectrogram = librosa.amplitude_to_db(spectrogram)
                
                # Determine the label and create the directory
                label = self.df[self.target].iloc[global_idx]  # Use global_idx here instead of i
                folder_name = f'label_{label}'
                folder_path = os.path.join(path, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                
                # Save the spectrogram
                plt.figure(figsize=(2.24, 2.24), dpi=100)
                librosa.display.specshow(log_spectrogram, sr=sampling_rate, hop_length=self.hop_length)
                plt.axis('off')
                plt.savefig(
                    os.path.join(folder_path, f'spectrogram_{self.df.index[global_idx]}.png'),
                    bbox_inches='tight', pad_inches=0
                )
                plt.close()
                plt.clf()
    
            # Manually trigger garbage collection
            gc.collect()

    def mfcc(self, batch_size=10):
        mfcc_features = []
        labels = []

        # Process in batches
        for start_idx in range(0, len(self.df), batch_size):
            end_idx = min(start_idx + batch_size, len(self.df))
            for i in range(start_idx, end_idx):
                # Load the audio file
                audio_data, sampling_rate = librosa.load(
                    os.path.join(self.path, f'{self.df.index[i]}.wav'),
                    sr=None
                )
                
                # Compute the MFCC
                ind_mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mfcc=20), axis=1)
                mfcc_features.append(ind_mfcc)
                
                # Get the label
                label = self.df[self.target].iloc[i]
                labels.append(label)

            # Manually trigger garbage collection
            gc.collect()

        # Create a DataFrame with MFCC features
        mfcc_df = pd.DataFrame(mfcc_features, columns=[f'mfcc_{i}' for i in range(20)])
        mfcc_df['label'] = labels

        return mfcc_df


# In[ ]:


def split_into_separate_dataframes(df, chunk_size):
    """
        Splits a DataFrame into separate variables, each a DataFrame with a specified number of rows,
        and prints out the names of the new variables.
    
        Parameters:
        df (pandas.DataFrame): The original DataFrame to split.
        chunk_size (int): The number of rows each chunk should have.
    """
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)
    variable_names = []

    for i in range(num_chunks):
        chunk_df = df[i*chunk_size:(i+1)*chunk_size].copy()
        variable_name = f'train_df_{i+1}'
        globals()[variable_name] = chunk_df
        variable_names.append(variable_name)

    return variable_names

