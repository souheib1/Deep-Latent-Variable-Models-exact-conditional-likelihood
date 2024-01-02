import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from sklearn.metrics import f1_score

class Sampler:

    def __init__(self, vae_model, data, noisy_data, missing_data_indices, T, sampling_algorithm='pseudo_gibbs'):

        self.vae_model = vae_model
        self.data = data
        self.noisy_data = noisy_data
        self.missing_data_indices = missing_data_indices
        self.T = T
        self.sampling_algorithm = sampling_algorithm
        self.samples = [noisy_data.data.numpy().copy()]
        self.f1_score = []

    @torch.no_grad()
    def sample(self):
        if self.sampling_algorithm == 'pseudo_gibbs':
            self.sample_pseudo_gibbs()
        elif self.sampling_algorithm == 'metropolis_within_gibbs':
            self.sample_metropolis_within_gibbs()
        else:
            raise ValueError("Invalid sampling algorithm. Choose 'pseudo_gibbs' or 'metropolis_within_gibbs'.")

        return self.samples,self.f1_score


    def sample_pseudo_gibbs(self):
        x = self.noisy_data.clone()
        binary_data = self.data.clone().view(-1)
        for _ in range(1,self.T+1):
            z = self.vae_model.encode(x)
            estimated_x = self.vae_model.decode(z).view(-1)
            x = x.view(-1)
            x[self.missing_data_indices] = estimated_x[self.missing_data_indices]
            x[x < 0.5] = 0
            x[x >= 0.5] = 1
            binary_data[binary_data >= 0.5] = 1
            binary_data[binary_data < 0.5] = 0
            self.f1_score.append(f1_score(x,binary_data))
            x = x.view(1, 1, 28, 28)
            self.samples.append(x.data.numpy().copy())
            

    def sample_metropolis_within_gibbs(self):
        # TO DO
        return
  