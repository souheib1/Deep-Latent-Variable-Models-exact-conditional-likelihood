import torch
import torchvision
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torchvision import datasets, transforms
from sampler import Sampler
from variational_ae import VAE
import seaborn as sns 

sns.set_style("whitegrid")
np.random.seed(42)

def simulate_missing_data(image,fraction,type):
    
    imputed_image = image.clone()
    if missing_type == 'random':
        num_pixels = int(fraction * 28 * 28)
        pixel_indices = random.sample(range(28 * 28), num_pixels)
        imputed_image = imputed_image.view(-1)
        imputed_image[pixel_indices] = torch.rand((num_pixels,))
        imputed_image = imputed_image.view(1, 1, 28, 28)
  
    elif missing_type == 'half_top':
        imputed_image[:, :, :14, :] = torch.rand((1, 1, 14, 28))
        pixel_indices = torch.arange(0, 14 * 28)

    elif missing_type == 'half_bottom':
        imputed_image[:, :, 14:, :] = torch.rand((1, 1, 14, 28))
        pixel_indices = torch.arange(14 * 28, 28 * 28)

    else:
        raise ValueError("Invalid missing data type.")
    
    return imputed_image,pixel_indices

    
# Load the trained VAE model
vae_model = VAE()  # Replace with your actual trained VAE model
vae_model.load_state_dict(torch.load('../model/model_weights/VAE_MNIST_zdim_64_epochs_50.pth'))
vae_model.eval()

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='../mnist_data', 
                                train=False, 
                                transform=transform, 
                                download=True)
    
# Define the missing data scenarios
missing_data_scenarios = [
    {'fraction': 0.4, 'type': 'random'},
    {'fraction': 0.5, 'type': 'random'},
    {'fraction': 0.6, 'type': 'random'},
    {'fraction': 0.7, 'type': 'random'},
    {'fraction': 0.8, 'type': 'random'},
    {'fraction': 0.5, 'type': 'half_top'},
    {'fraction': 0.5, 'type': 'half_bottom'},
]

save_path ="./results/"
sampling_algorithm='pseudo_gibbs'
T=100
# Iterate over missing data scenarios
for scenario in missing_data_scenarios:
    fraction = scenario['fraction']
    missing_type = scenario['type']

    print (f"Senario {sampling_algorithm}: \n   fraction : {fraction} \n    type:{missing_type} ")

    # Generate the missing data scenario
    original_images = []
    binary_images = []
    noisy_images = []
    imputed_images = []
    f1_scores = []

    for i in range(25):
        image, label = test_dataset[i]
        image = image.unsqueeze(0)
                
        noisy_image,indices_to_remove = simulate_missing_data(image,fraction,missing_type)
        # Convert to binary
        binary_image = (image >= 0.5).float()

        # Create VaeSampler instance
        sampler = Sampler(vae_model, 
                          image, 
                          noisy_image, 
                          indices_to_remove, 
                          T=T,
                          sampling_algorithm=sampling_algorithm)

        # Perform sampling
        imputed_image, f1_score_values = sampler.sample()

        # Append to lists
        #original_images.append(image.squeeze().numpy())
        binary_images.append(binary_image.squeeze().numpy())
        noisy_images.append(noisy_image.squeeze().numpy())
        imputed_images.append(imputed_image[-1].squeeze())
        f1_scores.append(f1_score_values[-1])
        
    # Plot F1-score evolution for 1 image
    plt.figure(figsize=(10,5))
    plt.plot(range(1,T+1),f1_score_values)
    plt.xlabel("sampling iteration")
    plt.ylabel("F1-score")
    plt.title(f"Pseudo_Gibbs: Missing {fraction*100}% {missing_type}")
    plt.savefig(save_path+sampling_algorithm+"_F1evolution_"+str(fraction*100)+"_"+missing_type+".PNG")
    # Plot the images
    fig, axs = plt.subplots(3, 5, figsize=(20, 10))

    for j in range(5):
        # Plot binary image
        axs[0, j].imshow(binary_images[j], cmap='gray')
        axs[0, j].axis('off')
        axs[0, j].set_title('Binary original')

        # Plot missing image
        axs[1, j].imshow(noisy_images[j], cmap='gray')
        axs[1, j].axis('off')
        axs[1, j].set_title(f'Missing {fraction*100}% {missing_type}')

        # Plot reconstructed image
        axs[2, j].imshow(imputed_images[j], cmap='gray')
        axs[2, j].axis('off')
        axs[2, j].set_title(f'Reconstructed f1={f1_scores[j]:.2f}')

        plt.tight_layout()
        plt.savefig(save_path+sampling_algorithm+"_"+str(int(fraction*100))+"_"+missing_type+".PNG")
    print(f"average f1_score {np.mean(f1_scores):.2f}" )