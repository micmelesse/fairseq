import torch
from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt','data/wikitext-103/data-bin/wikitext-103')
assert isinstance(roberta.model, torch.nn.Module)