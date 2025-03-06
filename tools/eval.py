import argparse
import logging
import json
from pathlib import Path

import torch
from src.utils.config import Config
from src.utils.utils import setup_logging
from src.data.processor import DataProcessor
from src.models import build_oracle_net
from src.engine.evaluator import build_evaluator
from src.visualization.visualizer import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate OracleNet')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='eval_results',
                        help='directory to save evaluation results')
    return parser.parse_args()

def main():
    
    args = parse_args()
    
 
    setup_logging()
    
  
    config = Config()
    
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
   
    data_processor = DataProcessor(config)
    _, _, _, target_test_loader = data_processor.get_dataloaders()
    
   
    model = build_oracle_net(config)
    evaluator = build_evaluator(config, model)
    
    
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    accuracy, predictions, ground_truth = evaluator.evaluate(target_test_loader)
    analysis_results = evaluator.analyze_results(predictions, ground_truth)
    
   
    results = {
        'accuracy': float(accuracy),
        'class_accuracies': {int(k): float(v) for k, v in analysis_results['class_accuracies'].items()},
        'best_class': (int(analysis_results['best_class'][0]), float(analysis_results['best_class'][1])),
        'worst_class': (int(analysis_results['worst_class'][0]), float(analysis_results['worst_class'][1]))
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
  
    visualizer = Visualizer(output_dir)
    visualizer.plot_confusion_matrix(analysis_results['confusion_matrix'].numpy(),
                                   save_name='confusion_matrix.png')
    
    logging.info(f'Evaluation completed. Overall accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
