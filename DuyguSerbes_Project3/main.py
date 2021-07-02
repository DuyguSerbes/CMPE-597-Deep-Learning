from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import model
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import make_grid
import matplotlib
from torchvision.utils import save_image
import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import pandas as pd

# Create output folder to save reconstructed images
if not os.path.exists("./outputs"):
    os.makedirs("./outputs")

matplotlib.style.use('ggplot')

# Create a gif from reconstructed images
to_pil_image = transforms.ToPILImage()

def image_to_gif(images):
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave('./outputs/generated_images.gif', imgs)

# Save generated images
def save_generated_images(recon_images , epoch):
    save_image(recon_images.cpu(), "./outputs/epoch_"+ str(epoch) + ".jpg")


# Save and plot loss values of model
def save_loss(train_loss, valid_loss, train_bce_loss, test_bce_loss, train_kl_loss, test_kl_loss):
    dict = {'train_loss': train_loss, 'valid_loss': valid_loss, 'train_bce': train_bce_loss, 'valid_bce': test_bce_loss, 'train_kl': train_kl_loss, 'valid_kl': test_kl_loss}
    df = pd.DataFrame(dict)
    df.to_csv('./outputs/out.csv')
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='blue', label='train loss')
    plt.plot(valid_loss, color='blue', label='validation loss', linestyle='--')
    plt.title("MNIST VAE Loss ")
    plt.xlabel('Epochs')
    plt.legend()
    plt.ylabel('Loss')
    plt.savefig('./outputs/loss.jpg')
    plt.figure(figsize=(10, 7))
    plt.plot(train_bce_loss, color='red', label='train reconstruction term')
    plt.plot(test_bce_loss, color='red', label='validation reconstruction term', linestyle='--')
    plt.title("MNIST VAE Loss ")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./outputs/bce_loss.jpg')
    plt.figure(figsize=(10, 7))
    plt.plot(train_kl_loss, color='orange', label='train regularization term')
    plt.plot(test_kl_loss, color='orange', label='validation regularization term', linestyle='--')
    plt.title("MNIST VAE Loss ")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./outputs/kl_loss.jpg')
    #plt.show()



# Define GPU usage
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not train_on_gpu:
    print('CUDA is not available! Training on CPU...')
else:
    print('CUDA is available! Training on GPU...')

my_model = model.VAE().to(device)

# Define hyperparameters
sequence_length = 28
input_size = 28
patience=20
batch_size = 64
lr = 0.001
epochs = 100
valid_loss_min=100000000
optimizer = optim.Adam(my_model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

#Load dataset
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
valid_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
#Define data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)



#Check the improvement
grid_images = []
train_loss_list=[]
valid_loss_list=[]
train_kl_list=[]
train_bce_list=[]
valid_kl_list=[]
valid_bce_list=[]
#Main loop
for epoch in range(1, epochs + 1):
    print('Epoch: ', epoch)

    #Train mod
    my_model.train()

    # Follow accuracy and loss calues
    running_loss = 0.0

    running_kl = 0.0
    running_bce = 0.0
    for  batch_idx, (data, target) in tqdm(enumerate(train_loader), total=int(len(train_data) / train_loader.batch_size)):

        # GPU usage
        data= data.to(device)
        images = data.reshape(-1, sequence_length, input_size).to(device)
        # Fresh start for optimizer
        optimizer.zero_grad()
        # return reconstructed images, mean and log variance
        reconstruction, mean, variance= my_model(images)
        # Loss of mini-batch
        bce_loss = criterion(reconstruction, data)
        kl_term, loss = model.final_loss(bce_loss, mean, variance)
        # Backward
        loss.backward()
        # Update parameters
        optimizer.step()
        # Calculate training loss
        running_loss += loss.item()*data.size(0)
        running_kl += kl_term.item() * data.size(0)
        running_bce += bce_loss.item() * data.size(0)
    # Calculate total training loss
    train_loss = running_loss / len(train_loader.dataset)
    kl_loss= running_kl / len(train_loader.dataset)
    bce = running_bce / len(train_loader.dataset)
    train_loss_list.append(train_loss)
    train_bce_list.append(bce)
    train_kl_list.append(kl_loss)
    print("Train Loss: {:.4f}".format(train_loss))

    #Validation mode
    my_model.eval()

    running_loss = 0.0
    running_kl = 0.0
    running_bce = 0.0

    for  batch_idx, (data, target) in tqdm(enumerate(valid_loader),total=int(len(valid_data) / valid_loader.batch_size)):

        # GPU usage
        data= data.to(device)
        images =  data.reshape(-1, sequence_length, input_size).to(device)

        reconstruction, mean, variance= my_model(images)

        # Loss of mini-batch
        bce_loss = criterion(reconstruction, data)
        kl_term, loss = model.final_loss(bce_loss, mean, variance)

        # Calculate valid loss
        running_loss += loss.item()*data.size(0)

        #save last batch of generated images to check improvement
        if batch_idx == int(len(valid_data) / valid_loader.batch_size) - 1:
            recon_images = reconstruction
            actual_images = data

        # Calculate current loss
        running_kl += kl_term.item() * data.size(0)
        running_bce += bce_loss.item() * data.size(0)

    #Calculate overall loss
    kl_loss = running_kl / len(valid_loader.dataset)
    bce = running_bce / len(valid_loader.dataset)
    valid_loss = running_loss / len(valid_loader.dataset)
    valid_loss_list.append(valid_loss)
    valid_bce_list.append(bce)
    valid_kl_list.append(kl_loss)

    print("Valid Loss: {:.4f}".format(valid_loss))


    #Save the best model if validation loss has decreased
    #Early Stopping
    #If validation is not improved up to patient number, stop
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        print("best valid loss {:.6f}, train loss {:.6f}". format(valid_loss, train_loss))
        model_dir = 'model.pt'
        torch.save(my_model.state_dict(), model_dir)
        valid_loss_min = valid_loss
        no_improvement = 0
    else:
        no_improvement += 1
    if patience < no_improvement:
        print("Early stopping")
        break


    #Generate
    save_generated_images(recon_images, epoch)
    #save_generated_images(actual_images, epoch)
    image_grid = make_grid(recon_images.detach().cpu())
    grid_images.append(image_grid)

# Generate gif
image_to_gif(grid_images)
# Save the loss
save_loss(train_loss_list, valid_loss_list, train_bce_list, valid_bce_list, train_kl_list, valid_kl_list)
print('DONE')






