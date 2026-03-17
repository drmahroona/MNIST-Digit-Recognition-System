# 🎯 Accurate MNIST Digit Recognizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://digit-rec.streamlit.app/)

A high-accuracy handwritten digit recognition app built with **Streamlit** and **PyTorch**. Draw a digit or upload an image, and let a powerful CNN predict it – achieving >99% accuracy on the MNIST test set.

## Features

- ✏️ **Interactive Canvas**: Draw digits directly in your browser (white on black background).
- 📤 **Image Upload**: Support for PNG, JPG, JPEG, BMP (with optional color inversion).
- 🧠 **Advanced CNN Model**: 3 convolutional layers with batch normalization, dropout, and data augmentation.
- 📊 **Detailed Results**: 
  - Predicted digit and confidence score.
  - Probability distribution bar chart (using Plotly).
  - Top‑3 predictions.
  - Visualisation of the preprocessed image (28×28).
- ⚙️ **Customisable Settings**: Confidence threshold, preprocessing display, inversion toggle.
- 🏋️ **On‑demand Training**: Trains a model if none exists, with progress indicators and best‑model checkpointing.
- 🚀 **Optimised Inference**: Fast predictions with GPU support if available.

**Usage**

Draw a digit in the canvas area (use your mouse or touch screen).

Adjust stroke width in the sidebar if needed.

Click “Recognize Drawing” to see the prediction.

Upload an image of a handwritten digit via the file uploader.

If the digit is dark on a light background, enable the “Invert uploaded image colors” option.

View the result, confidence, probability chart, and preprocessed image.

Adjust the confidence threshold in the sidebar to filter low‑confidence predictions.

**Requirements**
See requirements.txt for the full list. Main dependencies:

streamlit – web framework

torch & torchvision – deep learning

pillow – image processing

numpy – numerical operations

plotly – interactive charts

streamlit-drawable-canvas – drawing canvas component
