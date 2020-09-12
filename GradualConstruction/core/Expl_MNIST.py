
import torch.nn as nn
from torchvision.utils import save_image
from Base import *
import tqdm


class Expl_MNIST():
    def __init__(self,model_path,data_name,d,\
                 iter,lr,l2_coeff,target_class,tv_beta,tv_coeff):

        #load saved model (.pt)
        self.model_path = model_path
        self.model=load_model(self.model_path)

        # load image (mnist)
        self.data_path='examples/MNIST/'
        self.data_name=data_name
        self.org_data, self.org_data_tensor=load_data(self.data_path,self.data_name)

        # setting for mask
        self.d=d
        self.D=int(np.shape(self.org_data)[0])
        self.M=int(self.D/self.d)

        # setting for optimization
        self.lr = lr
        self.iter = iter
        self.l2_coeff = l2_coeff
        self.tv_beta=tv_beta
        self.tv_coeff=tv_coeff


        self.ref_dist_path = self.org_data_tensor
        self.target_class=target_class
    def build(self):
        #get gradient
        gradient,output_org,output_org_s,pred_org=gen_grad(self.org_data_tensor,\
                                                           self.model, self.target_class)

        print('output_org :{}'.format(torch_print(output_org)))
        print('output_org_s :{}'.format(torch_print(output_org_s)))
        print('pred_org :{}\n'.format(torch_print(pred_org)))

        #sorting gradient in descending order
        grad_imp_sort=grad_processing(gradient,self.d)

        mask=np.ones((self.M,self.M))
        color=np.zeros((self.M,self.M))
        mask_num=0

        color_ini=torch.rand((1,1,28,28))

        #used in loss function
        ref_mean,ref_std=ref_output(pred_org[self.target_class].cpu().numpy(),self.model,'MNIST_ref')
        ref_std=torch.sqrt(torch.sqrt(ref_std))
        #print(ref_std)
        print(ref_mean)


        while(1):
            check = False
            imp_max = grad_imp_sort[0,mask_num]
            imp_max_row = torch.div(imp_max, int(28 / self.d))
            imp_max_col = imp_max - imp_max_row * int(28 / self.d)

            mask[imp_max_row, imp_max_col] = 0
            mask_tensor = numpy_to_torch(mask,requires_grad=False)
            upsampled_mask = upsample(mask_tensor,28,method='near')

            color[imp_max_row, imp_max_col] = 1

            color_tensor = numpy_to_torch(color, requires_grad=False)
            upsampled_color = upsample(color_tensor, 28, method='near')
            upsampled_color = torch.where(upsampled_color == 0.0, upsampled_color, color_ini.cuda())
            upsampled_color = Variable(upsampled_color, requires_grad=True)
            optimizer = torch.optim.Adam([upsampled_color], lr=self.lr)

            l2_index = (upsampled_color != 0).nonzero()

            for i in tqdm.tqdm(range(self.iter)):
                l2_loss=0
                composite_tensor=self.org_data_tensor.mul(upsampled_mask)+\
                    upsampled_color.mul(1-upsampled_mask)

                output_comp=self.model(composite_tensor)   #size=[1,10]
                output_comp_s=torch.nn.Softmax(dim=-1)(output_comp)
                pred_comp=torch.argmax(output_comp)

                l2_loss= torch.dist(self.org_data_tensor, composite_tensor,2)
                # loss function

                """
                loss=-output_comp[0,pred_org[self.target_class]]+\
                     self.tv_coeff * tv_norm(composite_tensor, self.tv_coeff)\
                    +self.l2_coeff*l2_loss
                """
                

                loss=torch.mean(torch.abs(output_comp-ref_mean))+ \
                     self.tv_coeff * tv_norm(composite_tensor, self.tv_coeff)\
                    #+self.l2_coeff*l2_loss

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            # if torch.mean(torch.abs(output_comp - ref_mean)) <= 5:
            #     if output_comp_s[0,pred_org[self.target_class]] >= 0.5:
            #         check=True
            #         break
            # if check==True:
            #     break
            if output_comp_s[0,pred_org[self.target_class]] >= 0.90:
                break
            mask_num+=1
            if mask_num==self.M*self.M:
                return

        print("mask_num: {}".format(mask_num))
        arg_path=str(self.data_name)[:-4]
        os.system("mkdir -p result/MNIST/{}".format(arg_path))


        org_save_file="Org_class{}.png".format(torch_print(pred_org[0]))
        per_save_file="Per_class{}.png".format(str(torch_print(pred_comp)))
        delta_save_file="Delta.png"

        org_data = self.org_data_tensor
        per_data = composite_tensor
        delta_data=torch.abs(org_data-per_data)

        save_path='result/MNIST/'+str(self.data_name)[:-4]+'_to_'+str(torch_print(pred_comp))+'.png'
        save_image(torch.reshape(self.org_data_tensor, (28, 28)),'result/MNIST/{}/{}'\
                   .format(arg_path,org_save_file))
        save_image(torch.reshape(per_data, (28, 28)),'result/MNIST/{}/{}'.\
                   format(arg_path,per_save_file))
        save_image(torch.reshape(delta_data, (28, 28)),'result/MNIST/{}/{}'.\
                   format(arg_path,delta_save_file))
        print ('output_comp: {}'.format(torch_print(output_comp)))
        print ('output_comp_s: {}'.format(torch_print(output_comp_s)))
        print ('pred_comp: {}'.format(torch_print(pred_comp)))
        #
        #
        # """
        # plt.imshow(transforms.ToPILImage()(self.org_data_tensor))
        # plt.show()
        # self.org_data_tensor[0,imp_max_row*self.d:imp_max_row*self.d+self.d
        #     ,imp_max_col*self.d:imp_max_col*self.d+self.d]=1
        # plt.imshow(transforms.ToPILImage()(self.org_data_tensor))
        # plt.show()
        # """