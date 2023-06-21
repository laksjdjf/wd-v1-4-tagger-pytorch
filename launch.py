import pandas as pd
from utils.preprocess import preprocess_image
from vit.vit import ViT
import torch

vit = ViT(3,448,9083)
vit.load_state_dict(torch.load("wd-v1-4-vit-tagger-v2.ckpt"))
csv = pd.read_csv("selected_tags.csv")

def predict(image):
    image = preprocess_image(image)
    img = torch.tensor(image).permute(2,0,1).unsqueeze(0)
    out = vit(img)[0]
    return {csv["name"][i]:out[i].item() for i in range(9083)}

import gradio as gr
demo = gr.Interface(fn=predict, 
             inputs=gr.inputs.Image(type="pil"),
             outputs=gr.outputs.Label(num_top_classes=20),
             )
demo.launch(debug=True)