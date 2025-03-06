import torch

class Config:
 
    def __init__(self):
  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42
        
    
        self.data_root = "./data"
        self.image_size = 224
        self.batch_size = 64
        self.num_workers = 4
        
    
        self.num_classes = 241  
        self.hidden_dim = 256
        
      
        self.num_control_points = 19
        self.displacement = 17
        self.alpha1 = 0.53  # edge magnitude coefficient
        self.alpha2 = 0.62  # edge direction coefficient
        self.beta1 = 0.74   # local contrast coefficient
        self.beta2 = 0.43   # noise suppression coefficient
        
  
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.num_epochs = 90
        self.lr_step_size = 10000
        self.lr_gamma = 0.1

        self.lambda1 = 0.48  # structure loss weight
        self.lambda2 = 0.47  # texture loss weight
        self.lambda3 = 0.54  # gap loss weight
        self.lambda4 = 0.52  # category loss weight