"""
Created on Sun Oct  4 14:37:23 2020

@author: homayoun
"""
import torchvision.models as models
import torch
from torchvision import datasets, transforms as T

from PIL import Image
from io import BytesIO
import base64


def pil_image_to_base64(pil_image):
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue())


def base64_to_pil_image(base64_img):
    return Image.open(BytesIO(base64.b64decode(base64_img)))


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = T.Compose([
                       T.Resize((224,224)),
                       T.ToTensor(),
                       normalize])

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)