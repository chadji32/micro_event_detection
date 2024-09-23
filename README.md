Detecting Micro-Events in Sleep using a Transformer model

A non-contact system designed to detect and classify sleep-related
micro-events, such as apnea and hypopnea, using the TimeSformer model.
This project aims to assist in the diagnosis and monitoring of sleep
disorders by analysing video data captured during overnight sleep
studies.

Description

This project use the TimeSformer model, a transformer-based architecture
for video classification, to automatically detect micro-events during
sleep. By processing video recordings collected during overnight sleep
studies lasting 8-10 hours per patient, the system identifies key events
like apnea and hypopnea episodes, enabling the calculation of the
Apnea-Hypopnea Index (AHI). The process includes comprehensive data
pre-processing, model training, and testing to ensure accurate
classification.

The goal of the project is to develop a non-invasive, contactless method
for monitoring sleep patterns, which could potentially serve as an
alternative to traditional sleep study methods, such as polysomnography.
The repository includes the coding to reproduce the study, from raw data
processing to AHI calculation.

Dependencies

Libraries that need to be installed:

pip3 install torch torchvision torchaudio \--index-url
https://download.pytorch.org/whl/cu124 #It needs to be adjusted based on
the specifications of the computer it is running on pillow pandas numpy
openCV shutils glob2 matplotlib collection transformers scikit-learn
seaborn more-itertools tqdm

To run the codes in the folder bed_detection, the following files need
to be downloaded: yolov3.weights:
https://github.com/patrick013/Object-Detection\-\--Yolov3/blob/master/model/yolov3.weights
yolov3: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
coco.name:
https://github.com/pjreddie/darknet/blob/master/data/coco.names

Installing

Before running any code, ensure that the paths are correctly set based
on the dataset\'s location.

Executing program

Steps:

1\. Bed Detection: Run the code in the bed_detection folder to isolate
the bed area. For some videos, this may not work. In such cases,
manually run the code in the manual_masking folder to isolate the bed
area.

2\. Frame Extraction: Run the code in the extract_frames folder to
extract frames from the videos.

3\. AHI Calculation from PSG: Run cal_AHI_psg.py to extract the AHI for
all participants. This script requires the \"Flow Event.txt\" files for
each participant. Based on the extracted data, create the test dataset.

4\. Training the Model: Navigate to the Train_coding folder and run
run.sh to train the model. This script will execute all experiments and
save the models and logs for each experiment.

5\. Testing the Model: Navigate to the Test_coding folder and run run.sh
to test the models trained in step 4. The script will generate logs,
predictions, and ROC curve images.

6\. Final AHI Calculation: Run the code in the AHI_calculation folder to
calculate the AHI for all patients in the test dataset.
