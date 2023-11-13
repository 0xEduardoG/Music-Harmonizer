# Instrument-Classification

## Overview & Business Problem

In 2023, the integration of smart home technologies into daily life has become increasingly prevalent, with approximately 15% of households worldwide incorporating some form of smart home feature. This project aims to capitalize on this trend by developing a sophisticated feature for smart home devices. The core functionality of this feature is to detect when someone is playing a musical instrument, classify the instrument type, identify the musical genre, and recognize specific notes being played. The ultimate goal is to create an interactive environment that harmonizes with the musician, enhancing their playing experience.

The implementation of such technology marks a significant advancement in the application of Smart Homes in fostering musical skills. It serves not only as a tool for learning new instruments and songs but also offers real-time feedback on aspects like tempo, tuning, and musical expression. This technology could revolutionize musical education and practice by providing an immersive, responsive environment for musicians of all levels.

## Data Sources

Nsynth DataSet
For this project, we utilized the Nsynth DataSet, a comprehensive collection curated by Magenta. Key features of this dataset include:

Size and Diversity: The dataset encompasses over 350,000 musical instances, each being a 4-second audio clip. This extensive collection ensures a diverse range of instruments and sounds.
Metadata: Accompanying the audio files is a detailed JSON file providing metadata for each clip. This metadata includes information about the instrument type, pitch, and other relevant characteristics.
Quality and Format: The audio clips are of high quality, suitable for sophisticated audio processing and analysis.

## Methods

To achieve this, first we needed to import the data, which was a little challenging due to the technological limitations we had at the moment.

By doing some Exploratory Data Analysis we could see that the instrument family was really unbalanced. We downsampled it to be able to better train our model.

### MFCCs

Then, for the sake of trying to save on computing cappabilities, we extracted the Mel Frequencies Cepstrum Coefficients of the music instances and try to run a regular Machine learning model with it.

The results can be seen in the first model notebook, with an accuracy of around 35%. it wasn't enough for what we were trying to achieve, we an established threshold of 95%.

### Spectrograms & Results

Since this approach didn't gave us the expected results, we decided to take a different approach, applying a Short Term Fourier Transformation on each audio file and then creating spectrograms with them.

With the spectrograms, we can apply a Convolusional Neural Network to classify them in a more powerful way, capturing more data than just the MFCCs.

The results were significantly better, with results of 98% on the training set and 99% on the test. Making it surpasing our desired threshold and therefore having the model saved and ready for deployment.

We wanted to take it a couple steps further, since the final goal is for it to recognize musical in a real environment, we recorded clips with differents instrument families (Piano, flute, strings, Vocal and Brass), and generated a Python Script that takes my file input (in m4a format), transforms it into a 4 second WAV file, and that is then transformed into an spectrogram, which is evaluated by our model. So once the script is done, the output is teh prediction of the model for each one of the four second chunks of the clip.

### Next Steps

* Deploy my first model into an app that allows me to record and predict automatically.
* Create a docker container to make it easier to deploy and solve any environment issue we might face.
* Create the genre classification model.
* Create a note classification model.
* Create an AI generated musical harmonizer.