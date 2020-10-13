"""
Created on Sat Oct  3 02:59:13 2020

@author: homayoun
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import torchvision.models as models
import torch
from utils import gram_matrix, transform, cos
import pandas as pd
import json

def search(query_pil,model, db='index.json'):
    if type(db) == str:
        with open(db) as json_file:
              data = json.load(json_file)
              db = pd.DataFrame(data)
    
    query = transform(query_pil.convert('RGB'))
    qFeatures = model(query.unsqueeze(0))
    qGram = gram_matrix(qFeatures).flatten()
    scores = db['Gram'].apply(lambda x: cos(torch.tensor(x),qGram).item())
    name  =  db['Name'][scores.argmax()]
    score = round(scores.max(),3)
    return name, score

query_folder = 'queries'
database_folder = 'database'
result_folder = 'results'
db_path = 'index.json'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)
    
plt.ioff()
if __name__ == "__main__":
    simNet = torch.nn.Sequential(*list(models.vgg16(pretrained=True).features.modules())[1:14])
    with open(db_path) as json_file:
              data = json.load(json_file)
              db = pd.DataFrame(data)
    queries_list = sorted(glob(os.path.join(query_folder,'*.jpg')) + glob(os.path.join(query_folder,'*.png')))
    if not queries_list:
        raise ValueError('the [{}] folder is not exist or is empty'.format(query_folder))
    for query_path in queries_list:
      query_pil = Image.open(query_path)
      name, score = search(query_pil, simNet)
      match_pil = Image.open(os.path.join(database_folder,name))
      fig,ax = plt.subplots(1,2)
      ax[0].imshow(query_pil)
      ax[1].imshow(match_pil)
      plt.savefig(format(os.path.join(result_folder,query_path.split('/')[-1])))
      
      print(query_path ,name, score)
      





