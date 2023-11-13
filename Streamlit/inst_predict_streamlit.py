import streamlit as st
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

model = load_model('data/my_model.h5')

def process_and_save_spectrograms(audio_file):
    
    n_fft = 2048
    hop_length = 512
    images = []

    if audio_file.name.endswith('.m4a'):
        audio_segment = AudioSegment.from_file(audio_file, format="m4a")
        buffer = BytesIO()
        audio_segment.export(buffer, format="wav")
        buffer.seek(0)
        # Load the m4a file data into librosa
        audio_data, sampling_rate = librosa.load(buffer, sr=None)
    else:
        audio_data, sampling_rate = librosa.load(audio_file, sr=None)
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
                
        # Create a BytesIO buffer to save the spectrogram image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
                
        # Load the image from the buffer
        images.append(Image.open(buf))
        

    return images

def preprocess_image(img, target_size):
    # Resize image
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    # Convert image to array
    img_array = np.array(img)
    # Ensure the image has 3 channels (in case it's black and white)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    # If the image has an alpha channel, remove it
    if img_array.shape[2] == 4:
        img_array = img_array[:,:,:3]
    # Convert the image data type to float32
    img_array = img_array.astype('float32')
    # Normalize the image
    img_array /= 255.0
    # Expand dimensions to fit model expected input
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, image):
    # Preprocess the image
    img_array = preprocess_image(image, target_size=(224, 224))  # Use 'image' instead of 'img'
    # Make a prediction
    prediction = model.predict(img_array, verbose=0)
    # Return the prediction
    return prediction

def decode_prediction(pred):
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
    # Assuming pred is a softmax output, get the index with the highest probability
    label_index = np.argmax(pred, axis=1)[0]
    # Map the index to a label
    label = label_dict[label_index]  # You need to define label_dict based on your model's classes
    return label

background_color = "#635c54"  # Replace with your desired background color
background_style = f"""
<style>
    .stApp {{
        background: {background_color};
    }}
</style>
"""

def add_custom_css():
    st.markdown("""
        <style>
        .header {
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

st.markdown(background_style, unsafe_allow_html=True)
st.title('Instrument Classification Model')
st.image('images/instruments.png')
add_custom_css()
st.markdown('<h1 class="header">Upload File down here</h1>', unsafe_allow_html=True)

col1, col2 = st.columns([1.1,4])
uploaded_files = col2.file_uploader('',type=['m4a','wav'])
spectrogram = col1.checkbox('Spectrogram')
prediction = col1.checkbox('Prediction')
st.audio(uploaded_files)
start_b = st.button('START')

if start_b:

    if uploaded_files is not None:
        processed_file = process_and_save_spectrograms(uploaded_files)

        if spectrogram:
            for img in processed_file:
                st.image(processed_file, use_column_width=True)
        if prediction:
            for img in processed_file:
                pred = predict(model,img)
                predicted_label = decode_prediction(pred)
                st.text(f"the answer is {predicted_label}")
