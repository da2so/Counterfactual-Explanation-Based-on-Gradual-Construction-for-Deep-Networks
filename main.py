from Expl_tabular import Expl_Tabular
from Expl_MNIST import Expl_MNIST
from Expl_text import Expl_text
from utils import class_to_name
import argparse

import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Counterfactual Explanations based on Gradual Construction')
    #MLP_explain_probSim_0_36.csv
    #sample9_0.png
    #chocolate_sauce2.jpg

    #12_coeff=0.3
    parser.add_argument('--data_name', type=str, default="noise.png", help='Input data')
    parser.add_argument('--l2_coeff', type=float, default=0.3, help='make mask smaller')
    parser.add_argument('--tv_beta', type=int, default=2, help='exponential number of total variation')
    parser.add_argument('--tv_coeff', type=float, default=4, help='make mask be more natural image')
    parser.add_argument('--lr', type=float, default=0.01, help='learnng rate')
    parser.add_argument('--iter', type=int, default=1000, help='iteration number')
    parser.add_argument('--target_class', type=int, default=6,help='Choose the class')

    parser.add_argument('--d', type=int, default='4',
                        help='determine size of mask')
    #saved_classifier/MLP_pytorch_HELOC_allRemoved.pt
    #saved_classifier/mnist_cnn.pt
    #saved_classifier/tut4-model.pt
    parser.add_argument('--model_path', type=str, default='saved_classifier/mnist_cnn.pt',
                        help='model_path')

    args = parser.parse_args()

    file_list=os.listdir("examples/MNIST")
    print(file_list)


    exp = Expl_MNIST(model_path=args.model_path, data_name=args.data_name \
                     , d=args.d, iter=args.iter, \
                     lr=args.lr, l2_coeff=args.l2_coeff, target_class=args.target_class, \
                     tv_beta=args.tv_beta, tv_coeff=args.tv_coeff)

    exp.build()
