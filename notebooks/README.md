# Notebook descriptions
| Notebook   | Description         |
|------------|---------------------|
| **ssl_mae_pretraining.ipynb** | Use this notebook to train the self-supervised masked autoencoder |
| **wsl_training.ipynb** | Use this notebook to train the semantic segmentation models. It can be used to train the model from scratch (baseline model) and to finetune the pretrained model |
| **wsl_hyperparameter_search.ipynb** | Use this notebook to perform a hyperparameter search for the semantic segmentation model |
| **visualize_data.ipynb** | The notebook allows visualizing and inspecting the input data (size, shape, spectral values, labels etc) |

# Recommended workflow
1. Run `ssl_mae_pretraining.ipynb` and save the pretrained model
2. Run `wsl_hyperparameter_search.ipynb` to identify the best set of hyperparameters
3. Run `wsl_training.ipynb`  
   3.1. Run and save the baseline model  
   3.2. Finetune the pretrained model
