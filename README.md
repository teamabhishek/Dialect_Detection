# End-to-end Dialect Identification (implementation on MGB-3 Arabic dialect dataset)
An End-to-End dialect identification on Arabic dialect Dataset. Used MFCCs and Mel-Spectrogram for the audio feature extraction of MGB-3 Arabic dialect dataset. Saved Extracted features in tfrecords files.

# Requirement
* Python, tested on 3.10.5
* Tensorflow, tested on 2.11.0
* python library sox, tested on 1.3.2
* python library librosa, tested on 0.10.0 

# Data list format
data_add consist of (location of wavfile) and (label in digit).

Example) "train.txt"
```
./data/train/EGY/EGY000001.wav 0
./data/train/EGY/EGY000002.wav 0
./data/train/NOR/NOR000001.wav 4
```

Labels of Dialect: 
- Egytion (EGY) : 0
- Gulf (GLF) : 1
- Levantine(LAV): 2
- Modern Standard Arabic (MSA) : 3
- North African (NOR): 4


# Model definition
Simple description of the model:

we used four 1-dimensional CNN (1d-CNN) layers (40x5 - 500x7 - 500x1 - 500x1 filter sizes with 1-2-1-1 strides and the number of filters is 500-500-500-3000) and two FC layers (1500-600) that are connected with a Global average pooling layer which averages the CNN outputs to produce a fixed output size of 3000x1. 



