#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pathlib
import torch
from fastai.vision.all import *
import argparse
from PIL import Image

# Force Windows to use WindowsPath instead of PosixPath
pathlib.PosixPath = pathlib.WindowsPath  

# RE-DECLARE CUSTOM CLASSES OR FUNCTIONS USED IN TRAINING
class GetLabel:
    def getlabel(filename):
        return filename.split('-')[0]

def load_model(model_path):
    """Load the trained model from file with Windows path fix."""
    learn = load_learner(model_path, cpu=True)
    return learn

def predict_image(model, image_path):
    """Predict the class of an input image."""
    img = PILImage.create(image_path)
    pred, pred_idx, probs = model.predict(img)
    return pred, probs[pred_idx].item()

def main():
    parser = argparse.ArgumentParser(description='Test trained food classification model')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()

    model_path = r"C:\Users\13054\Dropbox\My PC (LAPTOP-LJK85NJQ)\Downloads\ScriptAI\export.pkl"
    model = load_model(model_path)

    pred, confidence = predict_image(model, args.image_path)
    print(f'Prediction: {pred}, Confidence: {confidence:.4f}')

if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




