
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable,grad

from utils import *

def ref_output(class_idx,model,data_name,sample_N=100,TEXT=None):
    base_path='examples/'+data_name+'/'
    class_path='sample'+str(class_idx)
    data_path=base_path+class_path

    file_list=os.listdir(base_path)

    total_file_num=0
    for file in file_list:
        if class_path in file:
            total_file_num+=1

    tmp, tmp_tensor = load_data(base_path, file,TEXT)
    class_num = np.size(np.squeeze(model(tmp_tensor).cpu().detach().numpy()))

    ext=file[-4:]
    rand_num=np.random.randint(total_file_num,size=(1,sample_N))
    output_arr = torch.zeros((class_num,sample_N))
    if cuda_available():
        output_arr=output_arr.cuda()

    with torch.no_grad():
        for i in range(sample_N):
            sample,sample_tensor=load_data(base_path, class_path + '_' + str(rand_num[0, i]) + ext,TEXT)

            output=model.forward(sample_tensor)
            output_arr[:, i] = output
    samples_mean=torch.mean(output_arr,dim=1)
    samples_std=torch.std(output_arr,dim=1)
    return samples_mean, samples_std


def gen_grad(org_data_tensor,model,target_class):
    org_data_var=Variable(org_data_tensor,requires_grad=True)

    output_org=model.forward(org_data_var)
    output_org_s=torch.nn.Softmax(dim=-1)(output_org)

    #output_org is the score before softmax!!!
    pred_org=torch.argsort(output_org,descending=True)
    pred_org=torch.squeeze(pred_org)
    output_org_tmp=torch.squeeze(output_org)

    assert target_class < output_org_tmp.size()[0]
    gradient=grad(outputs=output_org_tmp[pred_org[target_class]],inputs=org_data_var)[0]

    return gradient,output_org, output_org_s, pred_org

def grad_processing(gradient,d):
    grad_imp = torch.abs(gradient)
    if d!=1:
        m = torch.nn.AvgPool2d((d, d), stride=(d, d))
        # m = torch.nn.MaxPool2d((self.d, self.d), stride=(self.d, self.d))
        # gradient=torch.abs(gradient)
        grad_imp = m(grad_imp)
        grad_imp = torch.reshape(grad_imp, (1, int(28 * 28 / (d * d))))
        grad_imp= torch.abs(grad_imp)
    grad_imp_sort = torch.argsort(grad_imp, descending=True)

    return  grad_imp_sort


