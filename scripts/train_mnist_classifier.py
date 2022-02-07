#!/usr/bin/python3

import argparse
import logging
import os
import yaml

from core.model import MNISTNet
from core.dataset import dataset_fn
from utils.config import load_config


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Train MNIST classifier"
    )
    parser.add_argument(
        "--config_template", action="store", type=str, help="config file template", 
        default='./config/classification_mnist.yaml'
    )  
    parser.add_argument(
        "--out_dir", action="store", type=str, help="location of generated model", 
        default='./experiments_rebuttal/classification-models'
    )  
    parser.add_argument(
        "--seed", dest="seed", action="store", default=1000, type=int, help="",
    )
    parser.add_argument(
        "--remove_digit", dest="remove_digit", action="store", default=5, type=int, 
        help='digit to remove (default: keep all digits'
    )   
   
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    params = load_config(args.config_template)
        
    if args.remove_digit != None:
        # remove digit from dataloader
        params['dataset']['dl']['p']['sampling_weights'][args.remove_digit] = 0
        params['model']['n_outputs'] = 10 # TODO maybe switch back to 9
        
        task_classifier_path = params['model']['task_classifier_path']
        
        model_name = f'mnist_no{args.remove_digit}'    
    
    else: 
        model_name = f'mnist'

    params['model']['remove_idx'] = args.remove_digit

    params['model']['task_classifier_path'] = os.path.join(args.out_dir, f'{model_name}.pt')    
    
    config_file = os.path.join(args.out_dir, f'{model_name}.yaml')

    with open(config_file, 'w') as fhandle:
        yaml.dump(params, fhandle, default_flow_style=False)    
        
    ###############################################################################################################################
    # Data preparation
    ###############################################################################################################################

    dataloader = dataset_fn(seed=args.seed, params_dict=params['dataset'])

    ###############################################################################################################################
    # Prepare model and training
    ###############################################################################################################################
    
    if params['model']['task_classifier_type'] == 'mnist':
        # model = MNISTNet(n_outputs=params['model']['n_outputs'],
        #                  checkpoint_path=params['model']['task_classifier_path'],
        #                  download=False, remove_digit=args.remove_digit)
        model = MNISTNet(n_outputs=params['model']['n_outputs'],
                         checkpoint_path=params['model']['task_classifier_path'],
                         download=False)    
    else:
        raise NotImplementedError     
       
    model.train_model(dataloader_train=dataloader['train']['p'])     
    acc = model.eval_model(dataloader=dataloader['validation']['p'])

    logging.info(f'Val acc: {acc}')

    logging.info('done')



