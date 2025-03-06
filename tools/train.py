import argparse
import logging
from pathlib import Path

import torch
from src.utils.config import Config
from src.utils.utils import setup_logging, set_seed
from src.data.processor import DataProcessor
from src.models import build_oracle_net
from src.loss.losses import build_loss_computer
from src.engine.trainer import build_trainer
from src.visualization.visualizer import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Train OracleNet')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint for resume')
    return parser.parse_args()

def main():
   
    args = parse_args()
    
   
    setup_logging()
    
  
    config = Config()
    
  
    set_seed(config.seed)
    
    
    visualizer = Visualizer()
    
  
    data_processor = DataProcessor(config)
    source_train_loader, source_val_loader, \
    target_train_loader, target_val_loader = data_processor.get_dataloaders()
    
    
    model = build_oracle_net(config)
    loss_computer = build_loss_computer(config)
    
    
    trainer = build_trainer(
        config=config,
        model=model,
        loss_computer=loss_computer,
        source_train_loader=source_train_loader,
        target_train_loader=target_train_loader,
        source_val_loader=source_val_loader,
        target_val_loader=target_val_loader
    )
    
   
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
   
    trainer.train(config.num_epochs)
    
   
    visualizer.plot_training_curves(trainer.train_records)
    
if __name__ == '__main__':
    main()