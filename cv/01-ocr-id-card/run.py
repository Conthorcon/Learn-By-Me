import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp


CKP_PATH = "/home/truongkhanh/work/learn-by-me/cv/01-ocr-id-card/test-model/main/ckpt/seg/best_model.pth"
TEST_SAMPLE = "/home/truongkhanh/work/learn-by-me/cv/01-ocr-id-card/dataset/ABC/test_frames/image/image56.png"

DEVICE = "cpu"

def inference():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    model.load_state_dict(torch.load(CKP_PATH, weights_only=True))
    model.eval()

    img = cv2.imread(TEST_SAMPLE)

    img_cpy = img.copy()

    img_cpy = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2RGB)
    img_cpy = cv2.resize(img_cpy, (512, 512))

    x = img.transpose(2,0,1)/255.0
    x = torch.tensor(x).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)[0][0].cpu().numpy()
        pred = cv2.resize(pred, (img.shape[1], img.shape[0]))

    mask = (pred > 0.5).astype(np.uint8) * 255

    plt.imshow(mask, cmap='gray')
    plt.show()

    print("Completed")
    # cv2.imwrite("mask.png", mask)

if __name__ == "__main__":
    inference()