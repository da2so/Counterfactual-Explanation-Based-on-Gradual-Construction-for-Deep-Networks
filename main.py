import argparse
import os

import torch

from CF import CF_method
#./example/HELOC_ref/sample0_5.csv
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Counterfactual Explanations based on Gradual Construction')

    parser.add_argument('--dataset', type=str, default="imdb", help="choice: ['mnist','imdb','heloc']")
    parser.add_argument('--data_path', type=str, default="This film is boring", help='input data path')
    parser.add_argument('--l2_coeff', type=float, default=0.3, help='coefficient of the l2 regularization')
    parser.add_argument('--tv_beta', type=int, default=2, help='exponential number of total variation')
    parser.add_argument('--tv_coeff', type=float, default=4, help='coefficient of the TV regularization')
    parser.add_argument('--lr', type=float, default=0.01, help='learnng rate')
    parser.add_argument('--n_iter', type=int, default=500, help='iteration number')
    parser.add_argument('--target_class', type=int, default=1,help='Choose the target class')
    parser.add_argument('--target_prob', type=float, default=0.9,help='target probability of the target class')
    parser.add_argument('--d', type=int, default='1',help='determine size of mask')
    parser.add_argument('--model_path', type=str, default='./models/saved/tut4-model.pt',
                        help="choice=['mnist_cnn.pt',tut4-model.pt','MLP_pytorch_HELOC_allRemoved.pt', ] ")


    args = parser.parse_args()

    
    dataset_dict={
            'mnist':{'CF_method':'Expl_image', \
                    'ref_path': './ref_data/MNIST_ref/',\
                    'saved_path':'./result/MNIST/'},\

            'imdb': {'CF_method':'Expl_text', \
                    'ref_path': './ref_data/IMDB_ref/',\
                    'saved_path':'./result/IMDB/'},\

            'heloc': {'CF_method':'Expl_tabular',\
                    'ref_path': './ref_data/HELOC_ref/',\
                    'saved_path':'./result/HELOC/'},\

            'uci_credit_card': {'CF_method':'Expl_tabular',\
                                'ref_path': './ref_data/UCI_Credit_Card_ref/',\
                                'saved_path':'./result/UCI_Credit_Card/'},\
            }

    dataset=dataset_dict[args.dataset]

    CF_expl=CF_method(CF_method_name=dataset['CF_method'],\
                    model_path=args.model_path, \
                    data_path=args.data_path, \
                    d=args.d, \
                    n_iter=args.n_iter, \
                    lr=args.lr, \
                    l2_coeff=args.l2_coeff, \
                    target_class=args.target_class, \
                    tv_beta=args.tv_beta, \
                    tv_coeff=args.tv_coeff,\
                    ref_path=dataset['ref_path'],\
                    target_prob=args.target_prob,\
                    saved_path=dataset['saved_path']        
        )
    CF_expl.run()

