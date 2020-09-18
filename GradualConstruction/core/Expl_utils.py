import os
import numpy as np


import torch.nn as nn
from torch.autograd import Variable, grad
import torch

from GradualConstruction.utils import load_data,cuda_available


def gen_grad(org_data_tensor,model,target_class):
    org_data_var=Variable(org_data_tensor,requires_grad=True)

    output_org=model.forward(org_data_var)
    output_org_s=nn.Softmax(dim=-1)(output_org)

    #output_org is the score before softmax
    pred_org=torch.argsort(output_org,descending=True)
    pred_org=torch.squeeze(pred_org)
    pred_label=torch.argmax(output_org)
    output_org_tmp=torch.squeeze(output_org)

    assert target_class < output_org_tmp.size()[0]
    gradient=grad(outputs=output_org_tmp[pred_org[target_class]],inputs=org_data_var)[0]

    return gradient,output_org, output_org_s, pred_org,pred_label

def grad_processing(gradient,d=None,w=None):

    grad_imp = torch.abs(gradient)

    if d!=1:
        avgpool = torch.nn.AvgPool2d((d, d), stride=(d, d))
        grad_imp = avgpool(grad_imp)
        grad_imp = torch.reshape(grad_imp, (1, int(w * w / (d * d))))
        grad_imp= torch.abs(grad_imp)
    grad_imp_sort = torch.argsort(grad_imp, descending=True)

    return  grad_imp_sort


def ref_output(class_idx,model,ref_path,class_num,sample_N=50,TEXT=None):

    data_path=ref_path+'class'+str(class_idx)+'/'

    file_list=os.listdir(data_path)
    file_num=len(file_list)
    rand_num=np.random.randint(file_num,size=(1,sample_N))
    output_arr = torch.zeros((class_num,sample_N))
    if cuda_available():
        output_arr=output_arr.cuda()

    with torch.no_grad():
        for i in range(sample_N):
            _,sample_tensor=load_data(data_path+file_list[rand_num[0,i]],TEXT)
            output=model.forward(sample_tensor)
            output_arr[:, i] = output
    samples_mean=torch.mean(output_arr,dim=1)
    samples_var=torch.var(output_arr,dim=1)

    return samples_mean, samples_var