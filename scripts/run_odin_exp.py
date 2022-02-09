#!/usr/bin/python3

import os
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import griddata

import torch

from core.model import get_classification_model
from core.dataset import dataset_fn
from utils.config import load_config, create_exp_from_config
from utils.helpers import set_rcParams

from core import odin

def run_grid_search(dl_in, dl_out, model, temperatures, epsilons, num_img, results_gridsearch_csv):
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    columns = ['temperature', 'epsilon', 'method', 'rocauc', 'fpr95']
    df = pd.DataFrame(columns=columns)
    
    for temper in temperatures:
        for epsi in epsilons:
        
            df_in = odin.predict_scores(model, device, dl_in, epsi, temper, num_img)
            df_out = odin.predict_scores(model, device, dl_out, epsi, temper, num_img)
        
            print(f'-----------------------------------------------------')
            print(f'Hyperparams t={temper}, eps={epsi}')

            for method in ['base', 'odin']:
                roc_auc, fpr95 = odin.evaluate_scores(df_in[df_in['method'] == method]['score'], 
                                                df_out[df_out['method'] == method]['score'])

                row = {'temperature': temper, 'epsilon': epsi, 'method': method,
                    'rocauc': roc_auc, 'fpr95': fpr95}            
                df = df.append(row, ignore_index=True)
            
                print(f'{method} AUC: {roc_auc}, FPR95: {fpr95}')

    # validation results:        
    df.to_csv(results_gridsearch_csv)
        
def plot_gridsearch_results(df, temperatures, epsilons, log_dir):
    
    set_rcParams()
    
    X, Y = np.meshgrid(temperatures, epsilons)
    subset = df.loc[df['method'] == 'odin']

    for measure in ['rocauc', 'fpr95']:
        fig, ax = plt.subplots(figsize=(3, 3))

        grid_z0 = griddata(subset[['temperature', 'epsilon']], subset[measure], (X, Y), method='nearest')
        
        cmap = 'crest'
        if measure == 'rocauc':
            vmin, vmax = 0.5, 1.0
            vmin, vmax = None, None
            cmap = f'{cmap}_r'

        elif measure == 'fpr95':
            vmin, vmax = 0.0, 1.0
            vmin, vmax = None, None
        
        ax = sns.heatmap(grid_z0, annot=True, linewidths=.5, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel(r'temperature $\tau$')
        ax.set_xticklabels(temperatures)
        ax.set_ylabel(r'perturbation $\epsilon$')
        ax.set_yticklabels(epsilons)
        
        file_name = os.path.join(log_dir, f'ood_{measure}.pdf')
        fig.savefig(file_name)

def select_best_param(results_gridsearch_csv):
    
    gridsearch_df = pd.read_csv(results_gridsearch_csv)

    subset = gridsearch_df.loc[gridsearch_df['method'] == 'odin']
    best_row = subset[subset.fpr95 == subset.fpr95.min()]

    temper = best_row['temperature'].values[0]
    epsi = best_row['epsilon'].values[0]
    
    return temper, epsi
    
def eval_best_param(dl_in, dl_out, model, temper, epsi, results_csv):
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # num_img = len(dl_in.dataset)
    num_img = None

    df_in = odin.predict_scores(model, device, dl_in, epsi, temper, num_img)
    df_out = odin.predict_scores(model, device, dl_out, epsi, temper, num_img)

    columns = ['temperature', 'epsilon', 'method', 'rocauc', 'fpr95']
    df = pd.DataFrame(columns=columns)

    for method in ['base', 'odin']:
        roc_auc, fpr95 = odin.evaluate_scores(df_in[df_in['method'] == method]['score'], 
                                        df_out[df_out['method'] == method]['score'])

        row = {'temperature': temper, 'epsilon': epsi, 'method': method,
            'rocauc': roc_auc, 'fpr95': fpr95}            
        df = df.append(row, ignore_index=True)

    df.to_csv(results_csv)        

    return df

@torch.no_grad()
def predict(dataloader, model, laplace=False):

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.to(device)))
        else:
            py.append(torch.softmax(model(x.to(device)), dim=-1))

    return torch.cat(py).cpu().numpy()

def main(exp_dir, config_file, seed, run_gridsearch=True, run_plot=True, run_eval=True,
         temper=None, epsi=None):

    exp_name = create_exp_from_config(config_file, args.exp_dir)

    print(f'run ODIN for configuration: {exp_name}')

    # paths
    log_dir = os.path.join(exp_dir, exp_name)
    results_gridsearch_csv = os.path.join(log_dir, 'ood_gridsearch.csv')
    results_test_csv = os.path.join(log_dir, 'ood_test.csv')

    # hyperparam range:   
    temperatures = [1, 10, 100, 1000]
    epsilons = [0, 0.001, 0.002, 0.003, 0.004]

    ###############################################################################################################################
    # Data preparation
    ###############################################################################################################################
    
    params = load_config(config_file)

    dataloader = dataset_fn(seed=seed, params_dict=params['dataset'])

    model = get_classification_model(params['model'])  

    ###############################################################################################################################
    # Hyperparameter search and evaluation on test fold
    ###############################################################################################################################

    model.eval()
                  
    if not run_gridsearch and not os.path.exists(results_gridsearch_csv):
        raise ValueError('must run grid search.')
          
    if run_gridsearch:
        num_img = None
        dl_in = dataloader['validation']['p']
        dl_out = dataloader['validation']['q']
        run_grid_search(dl_in, dl_out, model, temperatures, epsilons, 
                    num_img, results_gridsearch_csv)
        
    if run_plot:
        df = pd.read_csv(results_gridsearch_csv)
        plot_gridsearch_results(df, temperatures, epsilons, log_dir)
      
    if run_eval:        
        #TODO: temper epsi as input params
        temper, epsi = select_best_param(results_gridsearch_csv)
        
        if temper != None and epsi != None:        
            pass
        elif os.path.exists(results_gridsearch_csv):
            temper, epsi = select_best_param(results_gridsearch_csv)
        else:
            raise ValueError('must run grid search or provide hyperparams.')
        
        eval(dataloader, model, temper, epsi, results_test_csv) 
    
def eval(dataloader, model, temper, epsi, results_test_csv):

    dl_in = dataloader['test']['p']
    dl_out = dataloader['test']['q']
        
    df = eval_best_param(dl_in, dl_out, model, temper, epsi, results_test_csv)

    run_la_redux = True
    if run_la_redux:
        from laplace import Laplace
    
        print('--------------------------')
        
        la = Laplace(model, 'classification',
            subset_of_weights='last_layer',
            hessian_structure='kron')
        la.fit(dataloader['train']['p'])
        la.optimize_prior_precision(method='marglik')
        probs_laplace_id = predict(dataloader['test']['p'], la, laplace=True)
        probs_laplace_ood = predict(dataloader['test']['q'], la, laplace=True)
        x = probs_laplace_id.max(-1)
        y = probs_laplace_ood.max(-1)
        roc_auc, fpr95 = odin.evaluate_scores(x, y)        
        print(f'laplace -- AUC: {roc_auc}, FPR: {fpr95}, detection rate: {1-fpr95}')

        row = {'temperature': np.nan, 'epsilon': np.nan, 'method': 'laplace',
            'rocauc': roc_auc, 'fpr95': fpr95}            
        df = df.append(row, ignore_index=True)

        df.to_csv(results_test_csv)  
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run single ODIN experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp_dir", action="store", type=str, help="experiment folder", default='./experiments_rebuttal/individual-ood'
    )
    parser.add_argument(
        "--config_file", action="store", type=str, help="config file", default='./config/odin_mnist_no5_10outputs.yaml'
        # "--config_file", action="store", type=str, help="config file", default='./config/odin_camelyon_no3.yaml'
    )  
    parser.add_argument(
        "--seed", dest="seed", action="store", default=1000, type=int, help="random seed",
    )
    parser.add_argument('--run_gridsearch', default=True, type=bool, help='gridsearch flag')
    parser.add_argument('--run_plot', default=True, type=bool, help='plot flag')
    parser.add_argument('--run_eval', default=True, type=bool, help='eval flag')

    parser.add_argument(
        "--temperature", dest="temperature", action="store", default=None, type=int, help="temperature for ODIN",
    )
    parser.add_argument(
        "--epsilon", dest="epsilon", action="store", default=None, type=float, help="epsilon for ODIN",
    )
   
    args = parser.parse_args()
             
    os.makedirs(args.exp_dir, exist_ok=True)
                       
    main(args.exp_dir, args.config_file, args.seed, run_gridsearch=args.run_gridsearch, 
         run_plot=args.run_plot, run_eval=args.run_eval, temper=args.temperature, epsi=args.epsilon)

    print('done')



