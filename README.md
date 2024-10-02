
---

# Emotion Detection using Deep Learning

## Overview

This project implements an Emotion Detection System using Convolutional Neural Networks (CNNs) with Keras and TensorFlow. The model is trained to recognize and classify emotions from facial images. The emotions detected are Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The system provides a user-friendly GUI that allows users to upload an image and receive real-time emotion predictions.

## Project Structure

The project consists of two main components:

1. **Model Creation and Training**
   - A script that defines and trains the CNN model to detect emotions from facial images. The model is saved in JSON format along with its weights in HDF5 format.

2. **Emotion Detection GUI**
   - A graphical user interface built with Tkinter that allows users to upload an image and display the detected emotion.

## Requirements

To run this project, ensure you have the following packages installed:

- Python 3.x
- TensorFlow (2.x or later)
- Keras
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Tkinter (included with standard Python installations)

You can install the required packages using pip:

```bash
pip install tensorflow keras opencv-python numpy matplotlib pillow
```

## Usage

### Model Creation and Training

1. **Prepare your dataset**: Organize your training and testing data into directories. Each emotion should have its own subdirectory containing the respective images. The structure should look like this:

   ```
   train/
       Angry/
       Disgust/
       Fear/
       Happy/
       Neutral/
       Sad/
       Surprise/
   test/
       Angry/
       Disgust/
       Fear/
       Happy/
       Neutral/
       Sad/
       Surprise/
   ```

2. **Train the model**: Run the `model_create.py` script to train the model. It will generate a model architecture JSON file and a weights file.

3. **Model files**: After training, you will have the following files:
   - `model_a.json`: Contains the model architecture.
   - `model_weights.weights.h5`: Contains the trained weights of the model.

### Emotion Detection GUI

1. **Launch the GUI**: Run the `emotion_detector_gui.py` script.

2. **Upload an image**: Click the "Upload Image" button to select an image file from your computer.

3. **Detect Emotion**: After uploading, click the "Detect Emotion" button to see the predicted emotion displayed on the GUI.

## Example Usage

- Upload an image of a face, and the system will output the emotion detected, such as "Happy," "Sad," etc.

## Acknowledgments

- This project leverages Keras for building the deep learning model and OpenCV for image processing.
- The Haar Cascade classifier is used for face detection.

## License

This project is licensed under the MIT License.

---
