import torch
import torchvision
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sampler import Sampler
from variational_ae import VAE
import seaborn as sns 
import argparse

sns.set_style("whitegrid")
np.random.seed(42)

def simulate_missing_data(image,fraction,missing_type,input_size=28):
    """
    Simulate missing data in an image based on the specified type and fraction.

    Parameters:
        image (torch.Tensor): Input image.
        fraction (float): Fraction of missing data to simulate.
        missing_type (str): Type of missing data ('random', 'half_top', or 'half_bottom').
        
    """
    
    imputed_image = image.clone()
    if missing_type == 'random':
        num_pixels = int(fraction * input_size * input_size)
        pixel_indices = random.sample(range(input_size * input_size), num_pixels)
        imputed_image = imputed_image.view(-1)
        imputed_image[pixel_indices] = torch.rand((num_pixels,))
        imputed_image = imputed_image.view(1, 1, input_size, input_size)
  
    elif missing_type == 'half_top':
        imputed_image[:, :, :input_size//2, :] = torch.rand((1, 1, input_size//2, input_size))
        pixel_indices = torch.arange(0, (input_size//2) * input_size)

    elif missing_type == 'half_bottom':
        imputed_image[:, :, input_size//2:, :] = torch.rand((1, 1, input_size//2, input_size))
        pixel_indices = torch.arange((input_size//2) * input_size, input_size * input_size)

    else:
        raise ValueError("Invalid missing data type.")
    
    return imputed_image,pixel_indices

    
def run_sampling(sampling_algorithm, vae_model, test_dataset, scenarios, save_path="./results/", T=100,input_size=28):
    """
    Run sampling for different missing data scenarios using the specified sampling algorithm.

    Parameters:
        sampling_algorithm (str): Sampling algorithm ('pseudo_gibbs' or 'metropolis_within_gibbs').
        vae_model (VAE): Trained Variational Autoencoder model.
        test_dataset: MNIST test dataset.
        save_path (str): Path to save the results.
        T (int): Number of sampling iterations.
    """

    for scenario in scenarios:
        fraction = scenario['fraction']
        missing_type = scenario['type']

        print(f"Scenario {sampling_algorithm}: \n   Fraction: {fraction} \n   Type: {missing_type} ")

        # Generate the missing data scenario
        binary_images = []
        noisy_images = []
        imputed_images = []
        f1_scores = []

        for i in range(5):
            image, _ = test_dataset[i]
            image = image.unsqueeze(0)

            noisy_image, indices_to_remove = simulate_missing_data(image, 
                                                                   fraction, 
                                                                   missing_type, 
                                                                   input_size=input_size)

            # Convert to binary
            binary_image = (image >= 0.5).float()

            # Create VaeSampler instance
            sampler = Sampler(vae_model, 
                              image, 
                              noisy_image, 
                              indices_to_remove, 
                              T=T, 
                              sampling_algorithm=sampling_algorithm,
                              input_size=input_size)

            # Perform sampling
            imputed_image, f1_score_values = sampler.sample()

            # Append to lists
            binary_images.append(binary_image.squeeze().numpy())
            noisy_images.append(noisy_image.squeeze().numpy())
            imputed_images.append(imputed_image[-1].squeeze())
            f1_scores.append(f1_score_values[-1])

        # Plot F1-score evolution for 1 image
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, T + 1), f1_score_values)
        plt.xlabel("Sampling Iteration")
        plt.ylabel("F1-score")
        plt.title(f"{sampling_algorithm.capitalize()}: Missing {fraction * 100}% {missing_type}")
        plt.savefig(save_path +"/"+ sampling_algorithm + "/"+sampling_algorithm + "_F1evolution_" + str(fraction * 100) + "_" + missing_type + ".PNG")

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
            axs[1, j].set_title(f'Missing {fraction * 100}% {missing_type}')

            # Plot reconstructed image
            axs[2, j].imshow(imputed_images[j], cmap='gray')
            axs[2, j].axis('off')
            axs[2, j].set_title(f'Reconstructed f1={f1_scores[j]:.2f}')

        plt.tight_layout()
        plt.savefig(save_path +"/"+ sampling_algorithm + "/"+ sampling_algorithm + "_" + str(int(fraction * 100)) + "_" + missing_type + ".PNG")

        print(f"    Average F1-score: {np.mean(f1_scores):.2f}")
        

def compare_sampling_algorithms(vae_model, test_dataset, scenarios, save_path="./results/", T=100,input_size=28):
    """
    Compare Pseudo-Gibbs and Metropolis-Within-Gibbs sampling algorithms for various missing data scenarios.

    Parameters:
        vae_model (VAE): Trained Variational Autoencoder model.
        test_dataset: test dataset.
        scenarios (list): List of dictionaries specifying missing data scenarios.
        save_path (str): Path to save the results.
        T (int): Number of sampling iterations.
    """
    random_scores_pg = {}
    random_scores_mwg = {}
    for scenario in scenarios:
        fraction = scenario['fraction']
        missing_type = scenario['type']

        print(f"Scenario: \n   Fraction: {fraction} \n   Type: {missing_type}")

        binary_images = []
        noisy_images = []
        imputed_images_pg = []
        imputed_images_mwg = []
        f1_scores_pg = []
        f1_scores_mwg = []

        for i in range(100,106):
            image, _ = test_dataset[i]
            image = image.unsqueeze(0)

            noisy_image, indices_to_remove = simulate_missing_data(image, fraction, missing_type,input_size=input_size)

            # Convert to binary
            binary_image = (image >= 0.5).float()

            # Create VaeSampler instance
            sampler_pg = Sampler(vae_model, 
                                 image, 
                                 noisy_image, 
                                 indices_to_remove, 
                                 T=T, 
                                 sampling_algorithm='pseudo_gibbs',
                                 input_size=input_size)
            
            sampler_mwg = Sampler(vae_model, 
                                  image, 
                                  noisy_image, 
                                  indices_to_remove, 
                                  T=T, 
                                  sampling_algorithm='metropolis_within_gibbs',
                                  input_size=input_size)

            # Perform sampling
            imputed_image_pg, f1_score_values_pg = sampler_pg.sample()
            imputed_image_mwg, f1_score_values_mwg = sampler_mwg.sample()

            # Append to lists
            binary_images.append(binary_image.squeeze().numpy())
            noisy_images.append(noisy_image.squeeze().numpy())
            imputed_images_pg.append(imputed_image_pg[-1].squeeze())
            imputed_images_mwg.append(imputed_image_mwg[-1].squeeze())
            f1_scores_pg.append(f1_score_values_pg[-1])
            f1_scores_mwg.append(f1_score_values_mwg[-1])

        # Plot F1-score evolution for 1 image
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, T + 1), f1_score_values_pg, label="Pseudo-gibbs", linestyle="--")
        plt.plot(range(1, T + 1), f1_score_values_mwg, label="Metroplis-Within-Gibbs", linestyle="-.")
        plt.xlabel("Sampling Iteration")
        plt.ylabel("F1-score")
        plt.title(f"Comparison: Missing {fraction * 100}% {missing_type}")
        plt.legend()
        plt.savefig(save_path + "comparison_F1evolution_" + str(fraction * 100) + "_" + missing_type + ".PNG")

        # Plot the images
        fig, axs = plt.subplots(4, 5, figsize=(20, 10))

        for j in range(5):
            axs[0, j].imshow(binary_images[j], cmap='gray')
            axs[0, j].axis('off')
            axs[0, j].set_title('Binary original')

            axs[1, j].imshow(noisy_images[j], cmap='gray')
            axs[1, j].axis('off')
            axs[1, j].set_title(f'Missing {fraction * 100}% {missing_type}')

            axs[2, j].imshow(imputed_images_pg[j], cmap='gray')
            axs[2, j].axis('off')
            axs[2, j].set_title(f'{"Pseudo-gibbs".capitalize()} Reconstruction f1={f1_scores_pg[j]:.2f}')
            
            axs[3, j].imshow(imputed_images_mwg[j], cmap='gray')
            axs[3, j].axis('off')
            axs[3, j].set_title(f'{"Metroplis-Within-Gibbs".capitalize()} Reconstruction f1={f1_scores_mwg[j]:.2f}')

        plt.tight_layout()
        plt.savefig(save_path + "comparison_" + str(int(fraction * 100)) + "_" + missing_type + ".PNG")

        mean_f1_score_pg = np.mean(f1_scores_pg)
        mean_f1_score_mwg = np.mean(f1_scores_mwg)
        if missing_type == "random":
            random_scores_pg[fraction * 100] = mean_f1_score_pg
            random_scores_mwg[fraction * 100] = mean_f1_score_mwg
    
    plt.figure(figsize=(10, 5))
    plt.plot(random_scores_pg.keys(), random_scores_pg.values(), label="Pseudo-gibbs", linestyle="--")
    plt.plot(random_scores_mwg.keys(), random_scores_mwg.values(), label="Metroplis-Within-Gibbs", linestyle="-.")
    plt.xlabel("Pourcentage of missing data")
    plt.ylabel("Average F1-score")
    plt.title(f"F1-score evaluation")
    plt.legend()
    plt.savefig(save_path + "F1evolution.PNG")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Inference parameters')
    parser.add_argument('--dataset', 
                    type=str, 
                    default="MNIST", 
                    help='Dataset for the training')
    
    parser.add_argument('--input_size', 
                    type=int, 
                    default=28, 
                    help='size in pixels of the input images')
        
    args = parser.parse_args()
    dataset = args.dataset
    input_size = args.input_size
    
    
    vae_model = VAE(input_size=input_size)  # Replace with your actual trained VAE model

    if dataset.upper() == "MNIST" :
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.MNIST(root='../datasets/mnist_data', 
                                        train=False, 
                                        transform=transform, 
                                        download=True)
        vae_model.load_state_dict(torch.load('../model/model_weights/VAE_MNIST_zdim_16_epochs_50.pth'))
    
    elif dataset.upper() == "OMNIGLOT" : 
            transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                            transforms.ToTensor(),
                                           transforms.Lambda(lambda x: 1-x)
                                           ])
            omniglot_dataset = torchvision.datasets.Omniglot(root='../datasets/omniglot_data',
                                                  background=False,
                                                  transform=transform,
                                                  download=True)

            # Split the dataset into training and testing sets
            train_size = int(0.7 * len(omniglot_dataset))
            test_size = len(omniglot_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(omniglot_dataset, [train_size, test_size])

            # Define batch size
            batch_size = 128
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)
            vae_model.load_state_dict(torch.load('../model/model_weights/VAE_OMNIGLOT_zdim_16_epochs_40.pth'))
    
    vae_model.eval()
        
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

    # print("run_sampling with pseudo_gibbs")
    # run_sampling(sampling_algorithm='pseudo_gibbs',
    #             vae_model=vae_model,
    #             test_dataset=test_dataset,
    #             scenarios=missing_data_scenarios,
    #             T=500,
    #             input_size=input_size)

    # print("run_sampling with metropolis_within_gibbs")
    # run_sampling(sampling_algorithm='metropolis_within_gibbs',
    #             vae_model=vae_model,
    #             test_dataset=test_dataset,
    #             scenarios=missing_data_scenarios,
    #             T=500,
    #             input_size=input_size)

    print("Compare the methods")
    compare_sampling_algorithms(vae_model=vae_model, 
                                test_dataset=test_dataset, 
                                scenarios=missing_data_scenarios, 
                                save_path="./results/", 
                                T=500,
                                input_size=input_size)