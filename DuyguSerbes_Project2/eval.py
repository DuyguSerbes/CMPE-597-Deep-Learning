import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import model
import sys


#Chech GPU availability
train_on_gpu = torch.cuda.is_available()



#Define parameters
#Number of subprocesses to use for data loading
num_workers = 0
#Batch size
batch_size = 32
#Criterion
criterion = torch.nn.CrossEntropyLoss()

#Evaluation function
def evaluate(test_loader,model_dir):
    #Load given model
    my_model = model.Model()
    my_model.load_state_dict(torch.load(model_dir))
    #Check GPU Usage
    if train_on_gpu:
        my_model.cuda()
    #Evaluation mode (important for batch normalization)
    my_model.eval()
    test_loss = 0.0
    test_total = 0.0
    test_correct = 0.0

    for data, target in test_loader:
        #GPU usage
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        #Forward
        output, _ = my_model(data)
        #Loss
        loss = criterion(output, target)
        #Loss value of mini*batch
        test_loss += loss.item() * data.size(0)
        #Get predicted class
        _, predicted = torch.max(output, 1)
        test_total += target.size(0)
        test_correct += (predicted == target).sum()


    #Calculate test accuracy
    correct = np.squeeze(test_correct.cpu().numpy())
    test_acc = 100 * correct / test_total

    #Calculate average test loss
    test_loss = test_loss / len(test_loader.dataset)

    #Print test loss and accuracy
    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('Test Acc: {:.6f}\n'.format(test_acc))



if __name__ == '__main__':

    #Test data load
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_data = datasets.CIFAR10('./data', train=False,
                                 download=True, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              num_workers=num_workers)

    try:
        model_dir = sys.argv[1]
        #print(model_dir)
    except:
        print('Please pass model directory')
        print('You should write in the command line: python3 eval.pt <model_directory>')
        print('For instance: python3 eval.py model.pt')
        exit()

    print(model_dir)
    evaluate(test_loader, model_dir)



