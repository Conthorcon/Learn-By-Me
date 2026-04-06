import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

import os

def predict():
    config = Cfg.load_config_from_name('vgg_transformer')

    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'

    detector = Predictor(config)

    for p in os.listdir("cropped_predictions"):
        # img = 'output_crops/5.jpg'
        path = os.path.join("cropped_predictions",p)
        img = Image.open(path)
        # plt.imshow(img)
        # plt.show()

        s = detector.predict(img)
        print(s)

if __name__ == "__main__":
    predict()
