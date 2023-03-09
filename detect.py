import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from mtcnn.core import image_tools
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face
from mtcnn.core.models import PNet,RNet,ONet,LossFn

def get_img(input_dir):
    xml_path_list = []
    for (root_path, dirname, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                xml_path = root_path + "/" + filename
                xml_path_list.append(xml_path)
    return xml_path_list


def detect_one(model, image_path, device, output_dir):

    img_array = cv2.imread(image_path)
    width = np.array(img_array).shape[1]
    height = np.array(img_array).shape[0]
    #
    # img_bg = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_array, (48, 48))
    img = image_tools.convert_image_to_tensor(img)
    img = torch.unsqueeze(img, 0)

    landmarks = model(img).detach().numpy()[0]
    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    tl = 2 or round(0.002 * (height + width) / 2) + 1

    for p in range(4):
        point_x = int(landmarks[p * 2] * width)
        point_y = int(landmarks[p * 2 + 1] * height)
        cv2.circle(img_array, (point_x, point_y), tl + 1, clors[p], -1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, os.path.split(image_path)[1])
    cv2.imwrite(save_path, img_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--input', type=str, default="")
    parser.add_argument('--output_dir', type=str, default="")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = args.model

    onet = ONet(is_train=False)
    onet.load_state_dict(torch.load(weights))
    img_path_list = get_img(args.input)

    for image_path in tqdm(img_path_list):
        detect_one(onet, image_path, device, args.output_dir)
