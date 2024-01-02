import numpy as np 
import torch
import matplotlib.pyplot as plt
from variational_ae import loss_vae,VAE
import argparse
import torchvision
from torchvision import datasets, transforms


def train_epoch(model, beta, criterion, optimizer, data_loader):
    
    train_loss_per_epoch = []
    model.train()
    for x_batch,_ in data_loader:
        x_batch = x_batch.to(device)    
        z_mean, z_log_var, reconstruction = model(x_batch)
        loss = criterion(beta=beta, 
                         original=x_batch.to(device).float(), 
                         z_mean=z_mean, 
                         z_log_var=z_log_var, 
                         reconstructed=reconstruction)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_per_epoch.append(loss.item())
    return np.mean(train_loss_per_epoch), z_mean, z_log_var, reconstruction


def eval_epoch(model, beta, criterion, optimizer, data_loader):
    val_loss_per_epoch = []
    model.eval()
    with torch.no_grad():
        for x_val, _ in data_loader:
            x_val = x_val.to(device)
            z_mean, z_log_var, reconstruction = model(x_val)
            loss = criterion(beta=beta, 
                             original=x_val.to(device).float(),
                             z_mean=z_mean, 
                             z_log_var=z_log_var, 
                             reconstructed=reconstruction)
            
            val_loss_per_epoch.append(loss.item())
    return np.mean(val_loss_per_epoch), z_mean, z_log_var, reconstruction


def train_VAE(model, train_loader, test_loader, beta, criteration = loss_vae, 
              learning_rate=1e-3, optimizer = None,
              epochs=15, batch_size=128,  plot_history=True, latent_dim=16,
              save_model=True, saving_path='./VAE_models/'):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    def __plot_history(history):
        plt.figure(figsize=(10, 5))
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        model_name = "VAE_MNIST_zdim_" + str(latent_dim)+"_epochs_"+str(epochs)
        plt.savefig(saving_path+ 'loss_'+model_name+'.png')
        plt.show()

    # Lists to store loss history for plotting
    train_loss_history = []
    val_loss_history = []

    # Training loop
    for epoch in range(epochs):
        
        # Train
        train_loss, z_mean, z_log_var, reconstruction = train_epoch(model, 
                                                                    beta=beta, 
                                                                    criterion=criteration, 
                                                                    optimizer=optimizer, 
                                                                    data_loader=train_loader)
        train_loss_history.append(train_loss/batch_size)                                            
        # Validation
        eval_loss, z_mean, z_log_var, reconstruction = eval_epoch(model, 
                                                                  beta=beta, 
                                                                  criterion=criteration, 
                                                                  optimizer=optimizer, 
                                                                  data_loader=test_loader)
        val_loss_history.append(eval_loss/batch_size) 
        
        eval_loss/=batch_size
        train_loss/=batch_size
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss :.4f} - Val Loss: {eval_loss:.4f}")

    if plot_history:
        history = {'loss': train_loss_history, 'val_loss': val_loss_history}
        __plot_history(history)

    if save_model:
        model_name = "VAE_MNIST_zdim_" + str(latent_dim)+"_epochs_"+str(epochs)
        torch.save(model.state_dict(), saving_path+model_name+".pth")

    return model

if __name__ == "__main__":   
    
    parser = argparse.ArgumentParser(description='Control the parameters of the model')
    
    parser.add_argument('--latent_dim', 
                        type=int, 
                        default=16, 
                        help='Dimension of the latent space')
    
    parser.add_argument('--beta', 
                        type=float, 
                        default=1.0, 
                        help='Beta value for Beta-VAE')
    
    parser.add_argument('--epochs', 
                        type=int, 
                        default=50, 
                        help='number of epochs for the training')
    
    parser.add_argument('--input_size', 
                        type=int, 
                        default=28, 
                        help='size in pixels of the input images')
    
    parser.add_argument('--saving_path', 
                        type=str, 
                        default='./model_weights/', 
                        help='path to save the model')

    args = parser.parse_args()
    beta = args.beta
    latent_dim = args.latent_dim
    epochs = args.epochs
    input_size = args.input_size
    saving_path = args.saving_path

    print("latent_dim set to ",latent_dim)
    print("beta set to ",beta)
    print("epochs set to ",epochs)
    print("input size set to ",input_size)
    print("saving_path set to ",saving_path)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    model = VAE().to(device)
     
    # Define a transform to preprocess the data
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./mnist_data', 
                                               train=True, 
                                               transform=transform, 
                                               download=True)
    
    test_dataset = torchvision.datasets.MNIST(root='./mnist_data', 
                                              train=False, 
                                              transform=transform, 
                                              download=True)

    # Create data loaders to handle batch processing
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False)

    model = train_VAE(model, 
                      beta=beta,
                      epochs= epochs, 
                      train_loader=train_loader,
                      test_loader=test_loader, 
                      optimizer=torch.optim.Adamax(model.parameters(), lr=1e-3),
                      criteration = loss_vae, 
                      save_model=True, 
                      saving_path=saving_path,
                      latent_dim=latent_dim)