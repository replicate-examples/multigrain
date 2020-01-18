import json
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torch.autograd import Variable
from multigrain.lib import get_multigrain

def setup():
    model = get_multigrain('resnet50')
    checkpoint = torch.load('data/joint_3BAA_0.5.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])

    with open('data/labels_map.txt') as fh:
        labels_map = json.load(fh)
        labels_map = [labels_map[str(i)] for i in range(1000)]

    return model, labels_map

def infer(data, image_path):
    model, labels_map = data
    tfms = transforms.Compose([transforms.ToTensor()])
    img = tfms(Image.open(image_path)).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output_dict = model(img)

    pred = output_dict["classifier_output"]

    result = {}

    for idx in torch.topk(pred, k=5).indices.squeeze(0).tolist():
        prob = torch.softmax(pred, dim=1)[0, idx].item()
        result[labels_map[idx]] = prob

    return result

