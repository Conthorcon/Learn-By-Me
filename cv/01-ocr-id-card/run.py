import cv2
import torch
from ultralytics import YOLO
import numpy as np

from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

import gradio as gr

from utils import *

config = Cfg.load_config_from_name('vgg_transformer')

config['cnn']['pretrained']=False
config['device'] = 'cuda:0'

model_detect = YOLO("/home/truongkhanh/work/learn-by-me/cv/01-ocr-id-card/test-model/main/ckpt/detect_text/best.pt")
model_rec = Predictor(config)


def process_image(image):
    cv2.imwrite("test.jpg", image)
    pred = model_detect("test.jpg")
    crop_imgs = split(pred[0], image)
    output = ""
    for img in crop_imgs:
        s = model_rec.predict(Image.fromarray(img))
        output += s + "\n"
    return output
    

def main():
    with gr.Blocks() as demo:
        input_image = gr.Image(label="Test Image")
        # detected_output = gr.Image(label="Detected Image")
        recog_output = gr.Textbox(label="Recognized Text")

        run_button = gr.Button("Run")

        run_button.click(
            fn=process_image,
            inputs=[input_image],
            # outputs=[detected_output, recog_output]
            outputs=[recog_output]
        )

    demo.launch()



if __name__ == "__main__":
    # image = cv2.imread("/home/truongkhanh/work/learn-by-me/cv/01-ocr-id-card/dataset/SROIE2019/train/images/X00016469620.jpg")
    # print(process_image(image))
    main()
            
