import matplotlib.pyplot as plt
import numpy as np
import torch
import model
from torchvision.utils import save_image
from torch.autograd import Variable
import sys
import os





def save_generated_images(recon_images, root, n=10):
    save_image(recon_images, root, nrow=n)

def grid_generator(vae, n=10):

    scale = 1.0

    # linearly spaced coordinates corresponding to the 2D plot of digit classes in the latent space
    latent_x = np.linspace(-scale, scale, n)
    latent_y = np.linspace(-scale, scale, n)
    latents = torch.FloatTensor(len(latent_y), len(latent_x), 2)

    for i, lx in enumerate(latent_x):
        for j, ly in enumerate(latent_y):
            latents[j, i, 0] = lx
            latents[j, i, 1] = ly
    latents = latents.view(-1, 2)  # flatten grid into a batch

    # reconstruct images from the latent vectors
    latents = latents
    image_recon = vae.decoder(latents)
    image_recon = image_recon
    save_generated_images(image_recon, "./generated_grid_images.jpg", n=10)


def random_generator(vae):
    n_samples = 10
    n_latent=2
    zsamples = Variable(torch.randn(n_samples * n_samples, n_latent))
    #zsamples = np.random.randn(100, 2)


    # reconstruct images from the latent vectors
    latents = torch.FloatTensor(zsamples)
    image_recon = vae.decoder(latents)
    image_recon = image_recon
    print(image_recon.shape)
    save_generated_images(image_recon, "./generated_random_images.jpg")
    if not os.path.exists("./generated_images"):
        os.makedirs("./generated_images")

    for i in range(image_recon.shape[0]):
        img=(image_recon[i, :, :, :])
        name="./generated_images/"+str(i)+".jpg"
        save_generated_images(img, name , n=1)

if __name__ == '__main__':
    model_dir = "model.pt"

    try:
        model_dir = sys.argv[1]
        #print(model_dir)
    except:
        print('Please pass model directory')
        print('You should write in the command line: python3 generator.py <model_directory>')
        print('For instance: python3 eval.py model.pt')
        print('If default model ./model.pt is available, generator will use it.')
        print('Else exit with an error...')

    vae = model.VAE()
    vae.load_state_dict(torch.load(model_dir))

    grid_generator(vae)
    random_generator(vae)