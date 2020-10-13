"""
Created on Sat Oct  3 03:12:59 2020

@author: homayoun
"""
import os
from glob import glob
from PIL import Image
import torchvision.models as models
import torch
from utils import gram_matrix, transform
import json
import base64
database = {'Name': [],
        'Gram': [],
        #'Image': [],
        }
database_folder = 'database'
if __name__ == "__main__":
    simNet = torch.nn.Sequential(*list(models.vgg16(pretrained=True).features.modules())[1:14])
    images_list = glob(os.path.join(database_folder,'*.jpg')) + glob(os.path.join(database_folder,'*.png'))
    
    for img_path in images_list:
      print(img_path)
      pil_img = Image.open(img_path)
      img = transform(pil_img)
      iFeatures = simNet(img.unsqueeze(0))
      iGram = gram_matrix(iFeatures).flatten()
      database['Name'].append(img_path.split('/')[-1])
      database['Gram'].append(iGram.detach().cpu().numpy().tolist())
      #database['Image'].append(base64.b64encode(pil_img.resize((224,224)).tobytes()).decode('UTF-8'))

    with open('index.json', 'w') as outfile:
      json.dump(database, outfile)