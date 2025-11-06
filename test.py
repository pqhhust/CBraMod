import torch
from datasets import pearl_dataset
from models import model_for_pearl
from finetune_evaluator import Evaluator

class Params:
    def __init__(self):
        self.datasets_dir = "/home/user01/aiotlab/pqhung/CBraMod/data/datasets/BigDownstream/pearl/notch_filtered"
        self.batch_size = 16
        self.use_pretrained_weights = True
        self.foundation_dir = "/home/user01/aiotlab/pqhung/CBraMod/pretrained_weights/pretrained_weights.pth"
        self.classifier = "all_patch_reps"
        self.dropout = 0.1
        self.model_dir = '/home/user01/aiotlab/pqhung/CBraMod/data/pearl/epoch20_acc_1.00000_pr_1.00000_roc_1.00000.pth'
        self.cuda = 0
        

params = Params()

loader = pearl_dataset.LoadDataset(params)
model = model_for_pearl.Model(params)

loaders = loader.get_data_loader()

model.load_state_dict(torch.load(params.model_dir, map_location='cpu'))

train_eval = Evaluator(params, loaders['train'])
val_eval = Evaluator(params, loaders['val'])
test_eval = Evaluator(params, loaders['test'])

print(train_eval.get_metrics_for_binaryclass(model))
print(val_eval.get_metrics_for_binaryclass(model))
print(test_eval.get_metrics_for_binaryclass(model))