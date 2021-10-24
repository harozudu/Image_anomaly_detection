from gan.model import DCGAN
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nb_gpu = 1 if device.type == 'cuda' else 0
model = DCGAN(ngpu=nb_gpu)
model.fit(
    epochs=1000,
    dataset_path="data/dice/"
)
