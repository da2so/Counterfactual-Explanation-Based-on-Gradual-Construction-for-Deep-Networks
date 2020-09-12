import torch.nn as nn
from Base import *
import sys
import os.path
import tqdm
def get_col_name(data_path, x):

    if 'heloc' in data_path or'HELOC' in data_path:
        label_name = 'RiskPerformance'
    elif 'UCI_Credit' in data_path:
        label_name = 'default.payment.next.month'
    else:
        print('label name not vaild!')
        sys.exit(0)
    features = [c for c in x.columns if c != label_name]

    return features, label_name


def data_info(model_path,data_name):
    if 'heloc' in model_path or 'HELOC' in model_path:
        data_path = 'examples/HELOC_test/'
        saved_path='result/HELOC/'+str(data_name)[:-4]+'/'
        ref_path='HELOC_test'
        if not os.path.isdir(saved_path):
            os.mkdir(saved_path)
        org_file='org.csv'
        comp_file='comp.csv'

    elif 'UCI_Credit' in model_path:
        data_path = 'examples/UCI_Credit_Card_test/'
        saved_path ='result/UCI_Credit_Card/'+str(data_name)[:-4]+'/'
        ref_path = 'UCI_Credit_Card_test'
        if not os.path.isdir(saved_path):
            os.mkdir(saved_path)
        org_file='org.csv'
        comp_file='comp.csv'

    else:
        print('label name not vaild!')
        sys.exit(0)

    return data_path,saved_path,org_file,comp_file,ref_path

class Expl_Tabular():
    def __init__(self,model_path,data_name,d,\
                 iter,lr,l2_coeff,target_class,tv_beta,tv_coeff):
        self.model_path = model_path
        self.model = load_model(self.model_path)
        self.data_name = data_name
        self.data_path, self.saved_path, \
        self.org_file, self.comp_file, self.ref_path = data_info(model_path, self.data_name)

        self.min, self.ran = tabular_data_info(model_path)

        data_load=pd.read_csv(self.data_path + self.data_name)
        self.features, self.label = get_col_name(self.data_path, data_load)

        self.org_data,self.org_data_tensor=load_data(self.data_path,self.data_name)

        assert d==1
        self.d=d
        self.D=int(np.shape(self.org_data)[0])
        self.M=int(self.D/self.d)

        self.lr=lr
        self.iter=iter
        self.l2_coeff=l2_coeff

        self.target_class=target_class


    def build(self):
        gradient, output_org, output_org_s, pred_org = gen_grad(self.org_data_tensor, \
                                                                self.model, self.target_class)
        print('org_data: {}'.format(torch_print(self.org_data_tensor)))
        print('output_org: {}'.format(torch_print(output_org)))
        print('output_org_s: {}'.format(torch_print(output_org_s)))
        print('pred_org :{}\n'.format(torch_print(pred_org)))

        grad_imp=torch.abs(gradient)
        grad_imp_sort=torch.argsort(grad_imp,0,descending=True)
        print('grad_imp_sort: {}'.format(torch_print(grad_imp_sort)))

        mask=np.ones(self.M)
        color = np.zeros(self.M)

        mask_num=0

        ref_mean,ref_std=ref_output(pred_org[self.target_class].cpu().numpy(),self.model,self.ref_path)
        ref_std=torch.sqrt(torch.sqrt(ref_std))
        #print(ref_std)
        print(ref_mean)

        while(1):
            check=False
            color[grad_imp_sort[mask_num]]=np.random.rand(1)
            color_tensor = numpy_to_torch(color,requires_grad=True)

            mask[grad_imp_sort[mask_num]]=0
            mask_tensor=numpy_to_torch(mask,requires_grad=False)

            optimizer = torch.optim.Adam([color_tensor], lr=self.lr)

            l2_index=(color_tensor != 0).nonzero()

            for i in tqdm.tqdm(range(self.iter)):
                l2_loss = 0
                composite_tensor = torch.add(self.org_data_tensor.mul(mask_tensor), \
                                             color_tensor.mul(1-mask_tensor))

                output_comp = self.model.forward(composite_tensor)
                output_comp=torch.squeeze(output_comp)
                output_comp_s=torch.nn.Softmax(dim=-1)(output_comp)

                l2_loss=torch.dist(self.org_data_tensor,composite_tensor,2)


                """
                loss=-output_comp_s[pred_org[self.target_class]]
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
            if output_comp_s[pred_org[self.target_class]] >= 0.5:
                break
            mask_num+=1
            print(composite_tensor)
            if mask_num==np.shape(self.org_data)[0]:
                return


        print("mask_num: {}\n".format(mask_num))
        print('comp_data: {}'.format(torch_print(composite_tensor)))
        print('output_comp: {}'.format(torch_print(output_comp)))
        print('output_comp_s: {}'.format(torch_print(output_comp_s))+'\n')

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