import torch.nn as nn
from Base import *
import sys
import os.path
import tqdm
import torch
from torchtext import data
from torchtext import datasets
import random
import numpy as np
import pickle

class Expl_text():
    def __init__(self,model_path,data_name,d,\
                 iter,lr,l2_coeff,target_class,tv_beta,tv_coeff):
        min_len=5
        with open('sentiment/TEXT.obj', 'rb') as f:
            self.TEXT = pickle.load(f)
        with open('sentiment/LABEL.obj', 'rb') as f:
            self.LABEL = pickle.load(f)


        self.model_path = model_path
        self.model = load_model(self.model_path,self.TEXT,self.LABEL)
        self.data_name = data_name
        # self.data_path, self.saved_path, \
        # self.org_file, self.comp_file, self.ref_path = data_info(model_path, self.data_name)

        self.data_path='Sentiment'
        self.org_data,self.org_data_tensor=load_data(self.data_path,self.data_name,self.TEXT)

        self.org_data_split=self.org_data.split(' ')

        assert d==1
        self.d=d

        if int(np.shape(self.org_data_split)[0]) >=min_len:
            self.D=int(np.shape(self.org_data_split)[0])
        else:
            self.D=min_len

        self.M=int(self.D/self.d)

        self.lr=lr
        self.iter=iter
        self.l2_coeff=l2_coeff

        self.target_class=target_class


    def build(self):
        print(self.org_data_tensor)
        org_data_var = Variable(self.model.embedding(self.org_data_tensor),requires_grad=True)
        output_org = self.model.expl(org_data_var)
        output_org_s =torch.sigmoid(output_org)


        pred_org = torch.argsort(output_org, descending=True)
        pred_org = torch.squeeze(pred_org)
        # output_org is the score before softmax!!!
        print(pred_org)
        gradient = grad(outputs=output_org, inputs=org_data_var)[0]

        print('org_data: {}'.format(torch_print(self.org_data_tensor)))
        print('output_org: {}'.format(torch_print(output_org)))
        print('output_org_s: {}'.format(torch_print(output_org_s)))
        grad_imp=torch.abs(gradient)
        grad_imp_sum=torch.sum(grad_imp,dim=-1)
        grad_imp_sort=torch.argsort(grad_imp_sum,-1,descending=True)
        print('grad_imp_sort: {}'.format(torch_print(grad_imp_sort)))

        mask=np.ones((self.M,100))
        color = np.zeros((self.M,100))

        mask_num=0

        proc_output=0 if output_org_s>=0.5 else 1
        target_prob=0.1 if output_org_s>=0.5 else 0.9
        target_cls=0 if output_org_s>=0.5 else 1

        ref_mean,ref_std=ref_output(proc_output,self.model,'Sentiment',10,self.TEXT)
        ref_std=torch.sqrt(torch.sqrt(ref_std))

        grad_imp_sort=torch.squeeze(grad_imp_sort)
        while(1):
            color[grad_imp_sort[mask_num].cpu(),:]=np.random.randn(1,100)
            color_tensor = numpy_to_torch(color,requires_grad=False)
            color_tensor=Variable(torch.squeeze(color_tensor,0),requires_grad=True)
            mask[grad_imp_sort[mask_num].cpu(),:]=0
            mask_tensor=numpy_to_torch(mask,requires_grad=False)
            mask_tensor=torch.squeeze(mask_tensor,0)

            optimizer = torch.optim.Adam([color_tensor], lr=self.lr)

            for i in tqdm.tqdm(range(self.iter)):
                l2_loss = 0
                composite_tensor = torch.add(org_data_var.mul(mask_tensor), \
                                             color_tensor.mul(1-mask_tensor))

                output_comp = self.model.expl(composite_tensor)
                output_comp=torch.squeeze(output_comp)
                output_comp_s=torch.sigmoid(output_comp)
                l2_loss=torch.dist(org_data_var,composite_tensor,2)
                """

                if target_cls==0:
                    loss=output_comp+self.l2_coeff*l2_loss
                elif target_cls==1:
                    loss=-output_comp+self.l2_coeff*l2_loss

                """
                # print(l2_loss)
                loss = torch.mean(torch.abs(output_comp - ref_mean))+self.l2_coeff*l2_loss


                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                # if output_comp_s[pred_org[self.target_class]] >= 0.5:
                #     check=True
                #     break
            # if check==True:
            #     break
            if target_cls==0 and target_prob >= output_comp_s:
                break
            elif target_cls==1 and target_prob <= output_comp_s:
                break
            mask_num+=1
            if mask_num==np.shape(self.org_data_split)[0]:
                return


        print("mask_num: {}\n".format(mask_num))
        #print('comp_data: {}'.format(torch_print(composite_tensor)))
        print('output_comp: {}'.format(torch_print(output_comp)))
        print('output_comp_s: {}'.format(torch_print(output_comp_s))+'\n')
        used_feat=list()
        for i in range(mask_num,-1, -1):
            used_feat.append(grad_imp_sort[mask_num])

        min_index=list()
        for i in range(len(used_feat)):
            used_feat_vec=composite_tensor[0,used_feat[i],:]
            min_val=10000000
            min_index.append(0)
            for j in range(np.shape(self.TEXT.vocab.vectors)[0]):
                compr=torch.tensor(self.TEXT.vocab.vectors[j,:]).cuda()
                dist=torch.dist(used_feat_vec,compr,2)
                if min_val > dist:
                    if self.TEXT.vocab.itos[j] == self.org_data_split[used_feat[i]]:
                        break
                    min_val=dist
                    min_index[i] = j

        composite_text=self.org_data_split
        for i in range(len(used_feat)):
            print(self.TEXT.vocab.itos[min_index[i]])
            composite_text[used_feat[i]]=self.TEXT.vocab.itos[min_index[i]]


        print('Original text: {}'.format(self.org_data))

        composite_text=' '.join(composite_text)
        print('Composite text: {}'.format(composite_text))





        """
        composite=np.squeeze(composite_tensor.cpu().detach().numpy())
        composite=np.reshape(composite,(1,len(composite)))

        feat = np.reshape(np.array(self.features), (len(self.features)))
        df=pd.DataFrame(data=composite,columns=feat)

        org_data=self.org_data_tensor.cpu().detach().numpy()
        org_data=np.squeeze(org_data)



        composite_denorm=composite*self.ran+self.min
        org_data_denorm=org_data*self.ran+self.min
        org_data_denorm=np.reshape(org_data_denorm,(1,np.shape(org_data_denorm)[0]))

        composite_denorm=np.around(composite_denorm)
        org_data_denorm=np.around(org_data_denorm)
        print('org_data: {}'.format(org_data_denorm))
        print('composite_data: {}'.format(composite_denorm))

        org_data=np.reshape(org_data,(1,np.shape(org_data)[0]))

        delta_data=np.abs(org_data-composite)

        df_org_res = pd.DataFrame(data=org_data_denorm, columns=feat)
        df_comp_res = pd.DataFrame(data=composite_denorm, columns=feat)

        df_org = pd.DataFrame(data=org_data, columns=feat)
        df_comp = pd.DataFrame(data=composite, columns=feat)
        df_delta = pd.DataFrame(data=delta_data,columns=feat)

        df_org_res.to_csv(self.saved_path+'res'+self.org_file,index=False)
        df_comp_res.to_csv(self.saved_path+'res'+self.comp_file,index=False)
        df_org.to_csv(self.saved_path+self.org_file,index=False)
        df_comp.to_csv(self.saved_path + self.comp_file, index=False)
        df_delta.to_csv(self.saved_path+'delta.csv',index=False)

        pred_org=torch.argmax(output_org_s)
        pred_comp=torch.argmax(output_comp_s)
        result='output_org: '+str(torch_print(output_org_s))+'\n'+\
        		'pred_org: '+str(torch_print(pred_org))+'\n'+\
               'output_comp: '+str(torch_print(output_comp_s))+'\n'+\
               'pred_comp: '+str(torch_print(pred_comp))
        f = open(self.saved_path+'result.txt', 'w')
        f.write(result)
        f.close()
        
        """