import os
import numpy as np

from torch.autograd import Variable,grad
import torch

from GradualConstruction.utils import torch_print, numpy_to_torch
from GradualConstruction.core.Expl_base import Expl_base
from GradualConstruction.core.Expl_utils import grad_processing, ref_output


class Expl_text(Expl_base):
    def __init__(self,model_path,data_path,d,n_iter,\
                lr,l2_coeff,target_class,target_prob,\
                tv_beta,tv_coeff,ref_path,saved_path):

        super(Expl_text, self).__init__(model_path, data_path, d, n_iter, lr, l2_coeff, \
                                           target_class, target_prob, tv_beta, tv_coeff, ref_path, saved_path)
        min_len=5
        self.org_data_split=self.org_data.split(' ')
        assert self.d==1
        if int(np.shape(self.org_data_split)[0]) >=min_len:
            self.D=int(np.shape(self.org_data_split)[0])
        else:
            self.D=min_len

        self.M=int(self.D/self.d)


    def build(self):

        org_data_var = Variable(self.model.embedding(self.org_data_tensor),requires_grad=True)
        output_org = self.model.expl(org_data_var)
        output_org_s =torch.sigmoid(output_org)

        pred_org = torch.argsort(output_org, descending=True)
        #pred_org = torch.squeeze(pred_org)

        gradient = grad(outputs=output_org, inputs=org_data_var)[0]

        proc_output=0 if output_org_s>=0.5 else 1
        self.target_prob=1- self.target_prob if output_org_s>=0.5 else self.target_prob
        target_cls=0 if output_org_s>=0.5 else 1
        org_class_name = 'Positive' if output_org_s>=0.5 else 'Negative'

        print(f"output for original data (before sigmoid) :{torch_print(output_org)}")
        print(f"output for original data :{torch_print(output_org_s)}")
        print(f"prediction label for original data :{org_class_name}")
        print('-'*40+'\n')
        class_num = len(pred_org)
        grad_imp=torch.abs(gradient)
        grad_imp_sum=torch.sum(grad_imp,dim=-1)
        grad_imp_sort=torch.argsort(grad_imp_sum,-1,descending=True).squeeze()

        mask=np.ones((self.M,100))
        color=np.zeros((self.M,100))
        mask_tensor = numpy_to_torch(mask,requires_grad=False)
        color_tensor = numpy_to_torch(color, requires_grad=False)
        mask_tensor = torch.squeeze(mask_tensor, 0)
        color_tensor=torch.squeeze(color_tensor, 0)
        mask_num=0

        ref_mean,ref_var=ref_output(proc_output,self.model,self.ref_path,class_num,\
                                    20,self.TEXT)

        used_feat = list()
        while(1):
            used_feat.append(grad_imp_sort[mask_num])

            color_tensor[:,grad_imp_sort[mask_num],:]=torch.rand(100)
            color_tensor=Variable(color_tensor,requires_grad=True)
            mask_tensor[:,grad_imp_sort[mask_num],:]=0


            optimizer = torch.optim.Adam([color_tensor], lr=self.lr)

            for i in range(self.n_iter):
                composite_tensor = torch.add(org_data_var.mul(mask_tensor), \
                                             color_tensor.mul(1-mask_tensor))

                output_comp = self.model.expl(composite_tensor)
                output_comp=torch.squeeze(output_comp)
                output_comp_s=torch.sigmoid(output_comp)
                l2_loss=torch.dist(org_data_var,composite_tensor,2)

                loss = torch.mean(torch.abs(output_comp - ref_mean))+self.l2_coeff*l2_loss


                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            print(f"Mask num {mask_num+1}\t Losses: total: {loss.item():3.3f}, \tl2: {l2_loss.item():3.3f}")

            if target_cls==0 and self.target_prob >= output_comp_s:
                break
            elif target_cls==1 and self.target_prob <= output_comp_s:
                break
            if mask_num==np.shape(self.org_data_split)[0]:
                return
            mask_num += 1


        per_class_name = 'Positive' if output_comp_s>=0.5 else 'Negative'
        print('\n'+'-' * 40)
        print(f"output for composite data (before sigmoid) :{torch_print(output_comp)}")
        print(f"output for composite data :{torch_print(output_comp_s)}")
        print(f"prediction label for composite data :{per_class_name}\n")


        min_index=list()
        for i in range(len(used_feat)):
            used_feat_vec=composite_tensor[0,used_feat[i],:]
            min_val=10000000
            min_index.append(0)
            for j in range(np.shape(self.TEXT.vocab.vectors)[0]):
                compr=self.TEXT.vocab.vectors[j,:].cuda()
                dist=torch.dist(used_feat_vec,compr,2)
                if min_val > dist:
                    if self.TEXT.vocab.itos[j] == self.org_data_split[used_feat[i]]:
                        break
                    min_val=dist
                    min_index[i] = j

        composite_text=self.org_data_split
        for i in range(len(used_feat)):
            composite_text[used_feat[i]]=self.TEXT.vocab.itos[min_index[i]]


        print('Original text: {}'.format(self.org_data))
        composite_text=' '.join(composite_text)
        print('Composite text: {}'.format(composite_text))

        self.saved_path = self.saved_path
        if not os.path.exists(self.saved_path):
            os.makedirs(self.saved_path)

        with open(self.saved_path+str(self.org_data)+'.txt', 'w') as out_file:
            out_file.writelines('Original text: {}\n'.format(self.org_data))
            out_file.writelines('Composite text: {}'.format(composite_text))

        out_file.close()