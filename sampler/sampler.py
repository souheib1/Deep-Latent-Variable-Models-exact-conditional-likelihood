import torch
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from variational_ae import loss_vae
class Sampler:

    def __init__(self, vae_model, data, noisy_data, missing_data_indices, T, sampling_algorithm='pseudo_gibbs',input_size=28):

        self.vae_model = vae_model
        self.data = data
        self.input_size = input_size
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
            x = x.view(1, 1, self.input_size, self.input_size)
            self.samples.append(x.data.numpy().copy())
            

    def sample_metropolis_within_gibbs(self): # à revoir
        x = self.noisy_data.clone()
        binary_data = self.data.clone().view(-1)
        z_current = torch.rand_like((self.vae_model.encode(x)))
        for _ in range(1,self.T+1):
            z = self.metropolis_within_gibbs_step(x,z_current)
            estimated_x = self.vae_model.decode(z).view(-1)
            x = x.view(-1)
            x[self.missing_data_indices] = estimated_x[self.missing_data_indices]
            x[x < 0.5] = 0
            x[x >= 0.5] = 1
            binary_data[binary_data >= 0.5] = 1
            binary_data[binary_data < 0.5] = 0
            self.f1_score.append(f1_score(x,binary_data))
            x = x.view(1, 1, self.input_size, self.input_size)
            self.samples.append(x.data.numpy().copy())
            z_current = z


    def metropolis_within_gibbs_step(self, current_x,z_current):
        current_z = z_current
        proposed_z = self.vae_model.encode(current_x)
        current_log_joint_prob = self.log_prob(current_x, current_z)
        proposed_log_joint_prob = self.log_prob(current_x, proposed_z)
        accept_prob = min(torch.exp(proposed_log_joint_prob - current_log_joint_prob).item(), 1.0)

        if torch.rand(1).item() <= accept_prob:
            return proposed_z
        else:
            return current_z

    def log_prob(self, x, z):
        decoded_z = self.vae_model.decode(z)
        z_mean, z_log_var, reconstruction = self.vae_model(decoded_z)
        return -torch.log(loss_vae(beta=1.0,
                                        original=x,
                                        z_mean=z_mean,
                                        z_log_var=z_log_var,
                                        reconstructed=decoded_z))

  