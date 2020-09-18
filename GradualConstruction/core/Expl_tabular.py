import numpy as np
import pandas as pd
import os

from torch.autograd import Variable
import torch

from GradualConstruction.utils import torch_print, tabular_data_info, get_col_name , numpy_to_torch
from GradualConstruction.core.Expl_base import Expl_base
from GradualConstruction.core.Expl_utils import gen_grad, grad_processing, ref_output



class Expl_tabular(Expl_base):
    def __init__(self,model_path,data_path,d,n_iter,\
                lr,l2_coeff,target_class,target_prob,\
                tv_beta,tv_coeff,ref_path,saved_path):

        super(Expl_tabular, self).__init__(model_path,data_path,d,n_iter,lr,l2_coeff,\
                target_class,target_prob,tv_beta,tv_coeff,ref_path,saved_path)

        self.min, self.ran = tabular_data_info(model_path)

        #data_load=pd.read_csv(self.data_path + self.data_name)
        self.features, self.label = get_col_name(self.data_path, self.org_data)
        assert d==1
        self.d=d

        self.D=int(np.shape(self.org_data)[1])
        self.M=int(self.D/self.d)




    def build(self):
        #get gradient
        gradient, output_org, output_org_s, pred_org,pred_label= gen_grad(self.org_data_tensor, \
                                                                self.model, self.target_class)
        
        print(f"orginal data: {torch_print(self.org_data_tensor)}")
        print(f"output for original data (before softmax) :{torch_print(output_org)}")
        print(f"output for original data :{torch_print(output_org_s)}")
        print(f"prediction label for original data :{torch_print(pred_label)}")
        print('-'*30+'\n')
        class_num=len(pred_org)
        #sorting gradient in descending order
        grad_imp_sort=grad_processing(gradient,self.d)

        mask=np.ones(self.M)
        color = np.zeros(self.M)
        mask_tensor = numpy_to_torch(mask,requires_grad=False)
        color_tensor = numpy_to_torch(color, requires_grad=False)
        mask_num=0

        #the logit scores of the training data
        ref_mean,ref_var=ref_output(pred_org[self.target_class].cpu().numpy(),\
            self.model,self.ref_path,class_num)

        while(1):

            color_tensor[:,:,grad_imp_sort[mask_num]]=torch.rand(1)
            color_tensor=Variable(color_tensor,requires_grad=True)
            mask_tensor[:,:,grad_imp_sort[mask_num]]=0

            optimizer = torch.optim.Adam([color_tensor], lr=self.lr)

            for i in range(self.n_iter):

                composite_tensor = torch.add(self.org_data_tensor.mul(mask_tensor), \
                                             color_tensor.mul(1-mask_tensor))

                output_comp = self.model.forward(composite_tensor)
                output_comp=torch.squeeze(output_comp)
                output_comp_s=torch.nn.Softmax(dim=-1)(output_comp)
                pred_label=torch.argmax(output_comp)

                l2_loss=torch.dist(self.org_data_tensor,composite_tensor,2)

                loss = torch.mean(torch.abs(output_comp - ref_mean))+self.l2_coeff*l2_loss
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            print(f"Mask num {mask_num+1}\t Losses: total: {loss.item():3.3f}, \tl2: {l2_loss.item():3.3f}")
            if output_comp_s[pred_org[self.target_class]] >= self.target_prob:
                break
            mask_num += 1
            if mask_num==self.M:
                print('Not found Counterfactual explanations')
                return

        print('\n'+'-'*40)
        print(f"composite data: {torch_print(composite_tensor)}")
        print(f"output for composite data (before softmax) :{torch_print(output_comp)}")
        print(f"output for composite data :{torch_print(output_comp_s)}")
        print(f"prediction label for composite data :{torch_print(pred_label)}\n")

        feat = np.reshape(np.array(self.features), (len(self.features)))
        
        composite=np.squeeze(composite_tensor.cpu().detach().numpy())
        composite=np.reshape(composite,(1,len(composite)))

        org_data=np.squeeze(self.org_data_tensor.cpu().detach().numpy())
        org_data=np.reshape(org_data,(1,len(org_data)))


        df_org = pd.DataFrame(data=org_data, columns=feat)
        df_comp = pd.DataFrame(data=composite, columns=feat)

        slash_idx=self.data_path.rfind('/')
        file_path=self.data_path[slash_idx:-4]
        self.saved_path=os.path.join(self.saved_path+file_path+'/')
        if not os.path.exists(self.saved_path):
            os.makedirs(self.saved_path)

        df_org.to_csv(self.saved_path+'org.csv',index=False)
        df_comp.to_csv(self.saved_path +'per.csv', index=False)

        pred_org=torch.argmax(output_org_s)
        pred_comp=torch.argmax(output_comp_s)
        result='output_org: '+str(torch_print(output_org_s))+'\n'+\
        		'pred_org: '+str(torch_print(pred_org))+'\n'+\
               'output_comp: '+str(torch_print(output_comp_s))+'\n'+\
               'pred_comp: '+str(torch_print(pred_comp))
        f = open(self.saved_path+'result.txt', 'w')
        f.write(result)
        f.close()