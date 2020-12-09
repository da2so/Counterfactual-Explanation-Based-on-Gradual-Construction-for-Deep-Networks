import numpy as np
import os

import torch
from torch.autograd import Variable
import torchvision.utils as vutils

from GradualConstruction.utils import torch_print, upsample, numpy_to_torch
from GradualConstruction.core.Expl_base import Expl_base
from GradualConstruction.core.Expl_utils import gen_grad, grad_processing,ref_output


def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] -img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))

    return (row_grad + col_grad)


class Expl_image(Expl_base):
    def __init__(self,model_path,data_path,d,n_iter,\
                lr,l2_coeff,target_class,target_prob,\
                tv_beta,tv_coeff,ref_path,saved_path):

        super(Expl_image, self).__init__(model_path,data_path,d,n_iter,lr,l2_coeff,\
                target_class,target_prob,tv_beta,tv_coeff,ref_path,saved_path)

        self.D = int(self.org_data_tensor.shape[3])
        self.M = int(self.D / self.d)
    def build(self):
        #get gradient
        gradient,output_org,output_org_s,pred_org,pred_label=gen_grad(self.org_data_tensor,\
                                                           self.model, self.target_class)

        print(f"output for original data (before softmax) :{torch_print(output_org)}")
        print(f"output for original data :{torch_print(output_org_s)}")
        print(f"prediction label for original data :{torch_print(pred_label)}")
        print('-'*40+'\n')
        #sorting gradient in descending order
        b,c,w,h=self.org_data_tensor.shape
        class_num=len(pred_org)
        grad_imp_sort=grad_processing(gradient,d=self.d,w=w)

        mask=np.ones((self.M,self.M))
        color=np.zeros((self.M,self.M))
        mask_tensor = numpy_to_torch(mask,requires_grad=False)
        color_tensor = numpy_to_torch(color, requires_grad=False)
        mask_num=0

        color_ini=torch.rand((b,c,w,h))

        #the logit scores of the training data
        ref_mean,ref_var=ref_output(pred_org[self.target_class].cpu().numpy(),\
            self.model,self.ref_path,class_num)

        while(1):
            imp_max = grad_imp_sort[0,mask_num]
            imp_max_row = int(torch.div(imp_max, int(w / self.d)))
            imp_max_col = imp_max - imp_max_row * int(w / self.d)
            
            mask_tensor[b-1,:,imp_max_row, imp_max_col] = 0
            upsampled_mask = upsample(mask_tensor,w,method='near')

            color_tensor[b-1,:,imp_max_row, imp_max_col] = 1

            upsampled_color = upsample(color_tensor, w, method='near')
            upsampled_color = torch.where(upsampled_color == 0.0, upsampled_color, color_ini.cuda())
            upsampled_color = Variable(upsampled_color, requires_grad=True)
            optimizer = torch.optim.Adam([upsampled_color], lr=self.lr)


            for i in range(self.n_iter):
                
                composite_tensor=self.org_data_tensor.mul(upsampled_mask)+\
                    upsampled_color.mul(1-upsampled_mask)

                output_comp=self.model(composite_tensor)   #size=[1,10]
                output_comp_s=torch.nn.Softmax(dim=-1)(output_comp)
                pred_label=torch.argmax(output_comp)

                l2_loss= torch.dist(self.org_data_tensor, composite_tensor,2)
                tv_loss= tv_norm(composite_tensor, self.tv_coeff)
                
                # loss function
                loss=torch.mean(torch.abs(output_comp-ref_mean))+ \
                    self.tv_coeff * tv_loss+\
                    self.l2_coeff*l2_loss

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            print(f"Mask num {mask_num+1}\t Losses: total: {loss.item():3.3f},\ttv: {tv_loss.item():3.3f} \tl2: {l2_loss.item():3.3f}")
            if output_comp_s[0,pred_org[self.target_class]] >= self.target_prob:
                break
            mask_num += 1
            if mask_num==self.M*self.M:
                print('Not found Counterfactual explanations')
                return


        slash_idx=self.data_path.rfind('/')
        file_path=self.data_path[slash_idx:-4]
        self.saved_path=self.saved_path+file_path+'/'
        if not os.path.exists(self.saved_path):
            os.makedirs(self.saved_path)

        org_save_file="Org_class{}.png".format(torch_print(pred_org[0]))
        per_save_file="Per_class{}.png".format(str(torch_print(pred_label)))

        vutils.save_image(self.org_data_tensor.data.clone(),self.saved_path+org_save_file)
        vutils.save_image(composite_tensor.data.clone(),self.saved_path+per_save_file)
        print('\n'+'-' * 40)
        print(f'output for composite data (before softmax) :{torch_print(output_comp)}')
        print(f'output for composite data :{torch_print(output_comp_s)}')
        print(f'prediction label for composite data :{torch_print(pred_label)}\n')