import pickle

from GradualConstruction.utils import load_data, load_model

class Expl_base():
    def __init__(self,model_path,data_path,d,n_iter,\
                lr,l2_coeff,target_class,target_prob,\
                tv_beta,tv_coeff,ref_path,saved_path):



        # setting for optimization
        self.lr = lr
        self.n_iter = n_iter
        self.l2_coeff = l2_coeff
        self.tv_beta=tv_beta
        self.tv_coeff=tv_coeff

        self.ref_path = ref_path
        self.saved_path=saved_path
        self.target_class=target_class
        self.target_prob=target_prob


        #load saved model (.pt)
        self.model_path = model_path
        self.data_path = data_path

        # setting for mask
        self.d=d

        # load input data and pre-trained model
        if 'IMDB' in ref_path:
            with open('./GradualConstruction/sentiment/TEXT.obj', 'rb') as f:
                self.TEXT = pickle.load(f)
            with open('./GradualConstruction/sentiment/LABEL.obj', 'rb') as f:
                self.LABEL = pickle.load(f)
            self.model = load_model(self.model_path,self.TEXT,self.LABEL )
            self.org_data, self.org_data_tensor = load_data(self.data_path,self.TEXT)
        else:
            self.model=load_model(self.model_path)
            self.org_data, self.org_data_tensor = load_data(self.data_path)

    
    def build(self):
        raise NotImplementedError