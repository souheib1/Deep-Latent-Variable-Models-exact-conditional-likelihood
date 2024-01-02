import torch
import torchvision
from torchvision import transforms
from variational_ae import VAE 
import matplotlib.pyplot as plt


def generate_samples(model, num_samples, latent_dim, device):
    # Generate random samples from the latent space
    with torch.no_grad():
        z_samples = torch.randn(num_samples, latent_dim).to(device)
        generated_images = model.decode(z_samples)
    return generated_images

def plot_generated_samples(samples, save_path=None,dataset="MNIST"):
    # Plot and save generated samples
    fig, axes = plt.subplots(1, len(samples), figsize=(len(samples) * 3, 3))
    for i, sample in enumerate(samples):
        axes[i].imshow(sample[0].cpu().numpy(), cmap='gray')
        axes[i].axis('off')
    if save_path:
        plt.savefig(save_path+dataset+"_samples")
    else:
        plt.show()

if __name__ == "__main__":
    
    # Load the trained model
    model = VAE()
    model.load_state_dict(torch.load('model_weights/VAE_MNIST_zdim_16_epochs_50.pth'))
    model.eval()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Generate and plot samples
    num_samples = 10  
    generated_samples = generate_samples(model, 
                                         num_samples,
                                         latent_dim=16,
                                         device=device)
    
    plot_generated_samples(generated_samples,
                           save_path='results/')
    
    model2 = VAE()
    model.load_state_dict(torch.load('model_weights/VAE_OMNIGLOT_zdim_16_epochs_55.pth'))
    model.eval()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Generate and plot samples
    num_samples = 10  
    generated_samples = generate_samples(model, 
                                         num_samples,
                                         latent_dim=16,
                                         device=device)
    
    plot_generated_samples(generated_samples,
                           save_path='results/',
                           dataset="OMNIGLOT")
    
    

