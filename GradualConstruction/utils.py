import torch
from torch.autograd import Variable
from torchvision import models
from torchvision.transforms.functional import normalize
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from model import Net,CNN
from PIL import Image
from torchvision.transforms import transforms
import pandas as pd
import sys
import spacy
from torchtext import data

def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda

def numpy_to_torch(data, requires_grad=True):
    if len(data.shape) < 3:
        output = np.float32([data])
    else:
        output= data
    output = torch.from_numpy(output)

    output.unsqueeze_(0)
    output=output.type(torch.FloatTensor)

    if cuda_available():
        output = output.cuda()

    v = Variable(output, requires_grad=requires_grad)
    return v


def load_model(model_path,TEXT=None,LABEL=None):
    #for saved model (.pt)
    if '.pt' in model_path:
        if torch.typename(torch.load(model_path)) == 'OrderedDict':
            if 'tut' in model_path:

                INPUT_DIM = len(TEXT.vocab)
                EMBEDDING_DIM = 100
                N_FILTERS = 100
                FILTER_SIZES = [3, 4, 5]
                OUTPUT_DIM = 1
                DROPOUT = 0.5
                PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

                model=CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
            elif 'mnist' in model_path:

                model=Net()
            model.load_state_dict(torch.load(model_path))

        else:
            model=torch.load(model_path)

    #for pretrained model
    elif model_path=='VGG19':
        model = models.vgg19(pretrained=True)
    elif model_path=='ResNet50':
        model = models.resnet50(pretrained=True)
    elif model_path=='DenseNet161':
        model=models.densenet161(pretrained=True)

    model.eval()
    if cuda_available():
        model.cuda()

    return model
def class_to_name(class_num):
    f = open('imagenet_class/imagenet1000_clsid_to_human.txt', 'r')

    while (1):
        line = f.readline()

        if line.find(str(class_num)) != -1:
            class_line = line
            break

    first_index = class_line.find("'")
    second_index = class_line[first_index + 1:].find("'") + first_index + 1

    class_name = class_line[first_index + 1:second_index]

    return class_name
def load_data(data_path,data_name,TEXT=None):
    org_path=data_path+data_name

    if 'MNIST' in data_path:
        org_data=Image.open(org_path).convert('L')

        trans = transforms.Compose([ \
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        org_data_tensor = trans(org_data).float()
        org_data_tensor = torch.reshape(org_data_tensor, (1, 1, 28, 28))  # size=[1,1,28,28] (input size)

    elif 'HELOC' in data_path or 'UCI' in data_path:
        org_data=pd.read_csv(org_path)
        org_data=np.squeeze(org_data.to_numpy())

        org_data_tensor=torch.from_numpy(org_data).float()
    elif 'Sentiment' in data_path:
        if 'examples' in data_path:
            org_data=np.loadtxt(org_path,dtype='str')
        else:
            org_data=data_name
        #need to download 'en' model
        nlp = spacy.load('en')

        if np.size(org_data) == 1:
            tokenized = [tok.text for tok in nlp.tokenizer(org_data)]
        else:
            tokenized = org_data
        if len(tokenized) < 5:
            tokenized += ['<pad>'] * (5- len(tokenized))
        org_data_tok = [TEXT.vocab.stoi[t] for t in tokenized]

        org_data_tensor = torch.LongTensor(org_data_tok)
        org_data_tensor = org_data_tensor.unsqueeze(0)


    elif 'ImageNet' in data_path:
        org_data=Image.open(org_path).convert('RGB')

        trans= transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),]
                                       )

        org_data_tensor=trans(org_data).float()
        org_data_tensor.unsqueeze_(0)
    if cuda_available():
        org_data_tensor=org_data_tensor.cuda()

    return org_data,org_data_tensor


def upsample(data,out_size,method='near'):
    if method == 'near':
        upsample= torch.nn.Upsample(size=(out_size, out_size), mode='nearest')
    elif method =='bilinear':
        upsample = torch.nn.UpsamplingBilinear2d(size=(out_size, out_size))

    if cuda_available():
        upsample=upsample.cuda()

    upsampled_data=upsample(data)

    return upsampled_data

def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] -img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))

    return (row_grad + col_grad)

def torch_print(input):
    input=torch.squeeze(input)
    output=np.round(input.cpu().detach().numpy(),3)

    return output

def heatmap(org_data_tensor, grad_tmp,num=0):
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )

    tmp1 = inv_normalize(torch.squeeze(org_data_tensor)).cpu().numpy().transpose(1, 2, 0)
    tmp2 = torch.squeeze(grad_tmp).cpu().detach().numpy()

    tmp2 = (tmp2 - np.min(tmp2)) / np.max(tmp2)
    tmp2 = 1 - tmp2
    tmp3 = cv2.applyColorMap(np.uint8(255 * tmp2), cv2.COLORMAP_JET)

    tmp4 = np.float32(tmp3) / 255
    tmp4 = 1.0 * tmp4 + tmp1

    path='result/examples/'+str(num)+'.png'
    cv2.imwrite(path, np.uint8(255 * tmp4))


    #cv2.imshow('image', tmp4)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def tabular_data_info(model_path):
    if 'heloc' in model_path or 'HELOC' in model_path:
        min = np.array([-8.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, -8.0, 0.0, \
                        2.0, 0.0, 0.0, 0.0, -8.0, 0.0, 0.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0])

        ran = np.array([797.0, 159.0, 254.0, 67.0, 14.0, 10.0, 100.0, 89.0, 9.0, 6.0, 83.0, \
                        16.0, 100.0, 32.0, 46.0, 46.0, 134.0, 173.0, 40.0, 21.0, 21.0, 108.0])

    elif 'UCI_Credit' in model_path:
        min = np.array([10000.0, 1.0, 0.0, 0.0, 21.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, \
                        -15308.0, -33350.0, -46127.0, -50616.0, -81334.0, -339603.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        ran = np.array([790000, 1, 6, 3, 54, 10, 9, 10, 10, 10, 10, 637057, 777320, \
                        605839, 667452, 612006, 853401, 493358, 1227082, 380478, \
                        400046, 388071, 403500])

    return min,ran


def get_discrete(ref_data):
    ref_dist = pd.read_csv(ref_data)
    ref_dist = np.squeeze(ref_dist.to_numpy())
    min, ran = tabular_data_info(ref_data)

    ref_dist = (ref_dist - min) / ran
    #ref_dist = torch.unique(ref_dist)

    check_discrete=list()
    discrete_dic={}
    for i in range(np.shape((ref_dist))[1]):
        unique_num=np.unique(ref_dist[:,i])
        if len(unique_num) <=10:
            check_discrete.append(i)
            discrete_dic[i]=unique_num

    return check_discrete,discrete_dic

def save_img(input,arg_path,file_path):
    input=np.array(input)
    fig=np.around((input+0.5)*255)
    fig=fig.astype(np.uint8).squeeze()
    pic=Image.fromarray(fig)
    pic.save("result/MNIST/{}/{}".format(arg_path,file_path))