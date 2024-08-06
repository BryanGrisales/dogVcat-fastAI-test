__all__ = ['is_cat', 'learn', 'classify_image', 'categories', 'image', 'label', 'intf']

from fastai.vision.all import *
import gradio as gr

def is_cat(x): 
    return x[0].isupper()

# Load the model
learn = load_learner('model.pkl')
categories = ('Dog', 'Cat')

def classify_image(img):
    # You might need to preprocess the image here if necessary
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

# Define the Gradio interface
image = gr.Image()
label = gr.Label()

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label)
intf.launch(inline=False)
