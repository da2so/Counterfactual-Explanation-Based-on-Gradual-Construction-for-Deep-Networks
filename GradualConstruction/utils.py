import numpy as np
from PIL import Image
import pandas as pd
import sys
import spacy

import torch
from torch.autograd import Variable
from torchvision import models
from torchvision.transforms.functional import normalize
from torchvision.transforms import transforms
from torchtext import data

from models.model import Net,CNN,MLP


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
            elif 'HELOC' or 'heloc' in model_path:
                input_size=22
                model=MLP(input_size)
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

def load_data(data_path,TEXT=None):

    if 'MNIST' in data_path:
        org_data=Image.open(data_path).convert('L')

        trans = transforms.Compose([ \
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        org_data_tensor = trans(org_data).float()
        org_data_tensor = torch.reshape(org_data_tensor, (1, 1, 28, 28))  # size=[1,1,28,28] (input size)

    elif 'HELOC' in data_path or 'UCI' in data_path:
        org_data=pd.read_csv(data_path)
        org_data_tmp=np.squeeze(org_data.to_numpy())

        org_data_tensor=torch.from_numpy(org_data_tmp).float()
    if TEXT != None:
        if '.txt' in data_path:
            org_data=np.loadtxt(data_path,dtype='str')
        else:
            org_data=data_path

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


    if cuda_available():
        org_data_tensor=org_data_tensor.cuda()

    return org_data, org_data_tensor


def upsample(data,out_size,method='near'):
    if method == 'near':
        upsample= torch.nn.Upsample(size=(out_size, out_size), mode='nearest')
    elif method =='bilinear':
        upsample = torch.nn.UpsamplingBilinear2d(size=(out_size, out_size))

    if cuda_available():
        upsample=upsample.cuda()
    upsampled_data=upsample(data)

    return upsampled_data


def torch_print(input):
    input=torch.squeeze(input)
    output=np.round(input.cpu().detach().numpy(),3)
    return output


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


def get_col_name(data_path, org_data):

    if 'heloc' in data_path.lower() :
        label_name = 'RiskPerformance'
    elif 'uci_credit' in data_path.lower:
        label_name = 'default.payment.next.month'
    else:
        print('label name not vaild!')
        sys.exit(0)
    features = [c for c in org_data.columns if c != label_name]

    return features, label_name

def save_img(input,arg_path,file_path):
    input=np.array(input)
    fig=np.around((input+0.5)*255)
    fig=fig.astype(np.uint8).squeeze()
    pic=Image.fromarray(fig)
    pic.save("result/MNIST/{}/{}".format(arg_path,file_path))