import numpy as np
from keras.models import load_model
import os
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Define appropiate paths
folder_path = r"C:\Users\eduar\Documents\Flatiron_DataScience\phase_5\Project\test_music"
model_path = r"C:\Users\eduar\Documents\Flatiron_DataScience\phase_5\Project\test_music\my_model.h5"
specs_path = r"C:\Users\eduar\Documents\Flatiron_DataScience\phase_5\Project\test_music\audio_processed"

# Define model
model = load_model(model_path)

# Create Function to process audio files and create spectrograms

def process_and_save_spectrograms(folder_path):
    
    n_fft=2048
    hop_length=512

    # Define the new folder path for processed audio
    processed_folder_path = os.path.join(folder_path, 'audio_processed')
    
    # Create the 'audio_processed' folder if it does not exist
    os.makedirs(processed_folder_path, exist_ok=True)

    # List all m4a files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.m4a'):
            # Construct full file path
            m4a_path = os.path.join(folder_path, filename)

            # Load the m4a file using librosa
            audio_data, sampling_rate = librosa.load(m4a_path, sr=None)

            # Skip the first second to avoid blank spaces when recording

            audio_data = audio_data[int(sampling_rate * 1):]

            # Process 4-second chunks
            for chunk_start in range(0, len(audio_data), int(4 * sampling_rate)):
                chunk_end = chunk_start + int(4 * sampling_rate)
                chunk_data = audio_data[chunk_start:chunk_end]
                
                # If the chunk is shorter than 4 seconds, discard it
                if len(chunk_data) < int(4 * sampling_rate):
                    continue

                # Compute the STFT and the spectrogram
                fourier_t = librosa.core.stft(chunk_data, n_fft=n_fft, hop_length=hop_length)
                spectrogram = np.abs(fourier_t)
                log_spectrogram = librosa.amplitude_to_db(spectrogram)
                
                # Save the spectrogram
                plt.figure(figsize=(2.24, 2.24), dpi=100)
                librosa.display.specshow(log_spectrogram, sr=sampling_rate, hop_length=hop_length)
                plt.axis('off')
                spectrogram_filename = f"{filename[:-4]}_spec_{chunk_start // sampling_rate}.png"
                plt.savefig(
                    os.path.join(processed_folder_path, spectrogram_filename),
                    bbox_inches='tight', pad_inches=0
                )
                plt.close()

# Run the function
process_and_save_spectrograms(folder_path)

def preprocess_image(img_path, target_size):
    # Load image
    img = tf.keras.utils.load_img(img_path, target_size=target_size, color_mode='rgb')
    # Convert image to array
    img_array = tf.keras.utils.img_to_array(img)
    # Expand dimensions to fit model expected input
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image
    img_array /= 255.0
    return img_array

def predict(model, img_path):
    # Preprocess the image
    img_array = preprocess_image(img_path, target_size=(224, 224))
    # Make a prediction
    prediction = model.predict(img_array, verbose=0)
    # Return the prediction
    return prediction

label_dict = {0:'Bass',
              1:'Brass',
              2:'Flute',
              3:'Guitar',
              4:'Keyboard',
              5:'Mallet',
              6:'Organ',
              7:'Reed',
              8:'String',
              9:'Synth_lead',
              10:'Vocal'
             }
    
for spectrogram_filename in os.listdir(specs_path):
    if spectrogram_filename.endswith('.png'):
        # Construct the full path to the spectrogram
        spectrogram_path = os.path.join(specs_path, spectrogram_filename)
        # Make a prediction
        pred = predict(model, spectrogram_path)
        label_index = np.argmax(pred)
        highest_probability = np.max(pred)
        # Output the label with the highest probability
        print(f"For the file {spectrogram_filename}: Label {label_dict[label_index]} with {highest_probability:.2f} probability.")
