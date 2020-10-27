# Counterfactual Explanation Based on Gradual Construction for Deep Networks
Counterfactual Explanation Based on Gradual Construction for Deep Networks Pytorch


## Requirements

- Pytorch 1.4.0 
- Python 3.6
- cv2 4.4.0
- matplotlib 3.3.1
- CUDA 10.1 (optional)
- spacy 2.2.3
- pandas 1.1.2
- torchtext 0.5.0

## Running the code

### MNIST dataset

Run the collowing command to get counterfactual explanation for MNIST data.

```shell
python main.py --dataset=mnist --model_path=./models/saved/mnist_cnn.pt --data_path=example/MNIST/0.png --d=4 --target_prob=0.9
```

Results are saved in **result** folder.


<img src="./assets/fig1.png" alt="drawing1" width="500"/>


### HELOC dataset

For IMDB dataset, you should download 'en' model. Type following command.

```shell
python -m spacy download en
```

Run the collowing command to get counterfactual explanation for IMDB data.


```shell
python main.py --dataset=imdb --model_path=./models/saved/tut4-model.pt --data_path="This film is good" --d=1 --target_prob=0.9
```
<img src="./assets/fig2.png" alt="drawing2" width="400"/>

### IMBD dataset

Run the collowing command to get counterfactual explanation for HELOC data.

```shell
python main.py --dataset=heloc --model_path=./models/saved/MLP_pytorch_HELOC_allRemoved.pt --data_path=./example/HELOC/1.csv --d=1 --target_prob=0.7
```

Target probability over 0.7 is not allowed because of pre-trained model capacity. 

<img src="./assets/fig3.png" alt="drawing3" width="400"/>


Arguments:

- `dataset` - Choose a experiment dataset 
	- choice: ['mnist','imdb','heloc','uci_credit_card'] 
- `data_path` - Input data (path)
- `l2_coeff` - Coefficient of the l2 regularization
- `tv_beta` - Exponential number of total variation (TV) regularization
- `tv_coeff` - Coefficient of the TV regularization
- `n_iter` - Iteration number
- `lr` - Learning rate
- `target_class` - Choose the target class 
	- 0: a class that has the first highest proability from original data
	- 1: a class that has the second highest proability from original data
- `target_prob` - Target probability of the target class
- `d` - Determine size of mask
- `model_path` - Saved model path 
	- choice=['mnist_cnn.pt',tut4-model.pt','MLP_pytorch_HELOC_allRemoved.pt'] 


## Understanding my paper

Check my blog!!
[HERE](https://da2so.github.io/2020-09-14-Counterfactual_Explanation_Based_on_Gradual_Construction_for_Deep_Networks/)