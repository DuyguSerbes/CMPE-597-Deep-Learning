import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import matplotlib.pyplot as plt
import model
import torch.optim as optim
import eval
import pandas as pd
from sklearn.manifold import TSNE


def tsne_results(matrix, targets, epoch):
    print(matrix.shape)
    print(targets.shape)
    plt.figure(figsize=(16, 10))
    #Color codes for 10 classes
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'brown', 'orange', 'purple'
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    tsne_results = tsne.fit_transform(matrix)
    # print(tsne_results)
    target_ids = range(len(classes))
    # put labels on the plot
    for i, c, label in zip(target_ids, colors, classes ):
        print(i, c, label)
        plt.scatter(tsne_results[targets == i, 0], tsne_results[targets == i, 1], c=c, label=label, alpha=0.2)
    name="Training Epoch="+  str(epoch)+  " t-SNE"
    plt.legend(classes)
    plt.title(name)
    plt.show()

# check CUDA availability
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not train_on_gpu:
    print('CUDA is not available! Training on CPU...')
else:
    print('CUDA is available! Training on GPU...')



#Define parameters for data loading
# number of subprocesses to use for data loading
num_workers = 0
#Batch Size
batch_size = 32
#Validation data percentage over all training data
valid_size = 0.2



#Data augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # p=probability of the image being flipped
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4, padding_mode='edge'),
    #Normalize images according to the mean and std values of images
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
#No data augmentation for validation and test data
valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#Load dataset
train_data = datasets.CIFAR10('./data', train=True,
                              download=True, transform=train_transform)
valid_data = datasets.CIFAR10('./data', train=True,
                              download=True, transform=valid_transform)
test_data = datasets.CIFAR10('./data', train=False,
                             download=True, transform=test_transform)

#Classes found in this dataset
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

#Suffle training data and select index to divide it as training and validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
print(split, "split")
train_idx, valid_idx = indices[split:], indices[:split]


#Define samplers
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

#Define data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
    num_workers=num_workers)


#Ccheck the improvement
train_acc_list=[]
train_loss_list=[]
valid_acc_list=[]
valid_loss_list=[]



my_model = model.Model()
print(my_model)

#GPU usage
if train_on_gpu:
    my_model.to(device)
    print("Transfering to cuda...")


#Define hyperparameters
#Learning rate
learning_rate = 0.001

#Number of epoch to be waited any improvement for early stopping
patience = 20

#Loss function
criterion = torch.nn.CrossEntropyLoss()

#Optimizer
optimizer = optim.Adam(my_model.parameters(), lr=learning_rate)

#Number of epoch
n_epochs = 150

#For best model criteria
valid_loss_min = np.Inf

#Follow no improvement epoch number
no_improvement=0

#Main loop
for epoch in range(1, n_epochs + 1):
    #Decreasing learning rate
    if epoch ==50:
        learning_rate=0.0001
    if epoch == 100:
        learning_rate = 0.00001
    #print(learning_rate, "learning rate")

    #Follow accuracy and loss calues
    train_loss = 0.0
    valid_loss = 0.0
    running_loss=0.0
    t_correct=0.0
    t_total=0.0

    #Train model
    my_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #GPU usage
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
        #Fresh start for optimizer
        optimizer.zero_grad()
        #Forward
        output , flatten = my_model(data)
        #Loss of mini-batch
        loss = criterion(output, target)
        #Backward
        loss.backward()
        #Update parameters
        optimizer.step()
        #Calculate training loss
        train_loss += loss.item() * data.size(0)
        #Calculate traininf accuracy
        _, predicted =torch.max(output, 1)
        t_total += target.size(0)
        t_correct += (predicted == target).sum()


        #IF YOU WANT TO SEE t-SNE PLOTS MAKE SURE TO COMMENT OUT OF BELOW LINES
        #t-SNE plots
        #if (epoch==1 or epoch==5 or epoch==10 or epoch==20 or epoch==30 or epoch==40 or epoch==50 or epoch==60 or epoch==70):
        #    flatten=np.squeeze(flatten.detach().cpu().numpy())
        #    tsne_target=np.squeeze(target.detach().cpu().numpy())

        #    if batch_idx==0  :
        #        matrix=flatten
        #        tsne_target_matrix=tsne_target
        #    else:
        #        matrix=np.append(matrix, flatten, axis=0)
        #        tsne_target_matrix=np.append(tsne_target_matrix,tsne_target, axis=0)
    #if (epoch==1 or epoch == 5 or epoch == 10 or epoch == 20 or epoch == 30 or epoch == 40 or epoch == 50 or epoch == 60 or epoch == 70):
        #tsne_results(matrix, tsne_target_matrix, epoch)


    #Calculate average training accuarcy and loss
    correct=np.squeeze(t_correct.cpu().numpy())
    train_acc=100 * correct / t_total
    train_loss = train_loss / len(train_loader.sampler)




    #Validation
    my_model.eval()
    v_total=0
    v_t_correct=0
    for batch_idx, (data, target) in enumerate(valid_loader):
        #GPU usage
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        #Forward
        output, _ = my_model(data)
        #Loss of mini-batch
        loss = criterion(output, target)
        #Calculate validation loss
        valid_loss += loss.item() * data.size(0)
        #Calculate validation accuracy
        _, predicted = torch.max(output, 1)
        v_total += target.size(0)
        v_t_correct += (predicted == target).sum()

    #Calculate average validation accuracy and loss
    v_correct = np.squeeze(v_t_correct.cpu().numpy())
    valid_acc=100 * v_correct / v_total
    valid_loss = valid_loss / len(valid_loader.sampler)


    #Print training and validation loss
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # Print training and validation accuracy
    print('Epoch: {} \tTraining Acc: {:.4f} \tValidation Acc: {:.4f}'.format(epoch, train_acc,valid_acc))

    #Trach the values
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)


    #Save the best model if validation loss has decreased
    #Early Stopping
    #If validation is not improved up to patient number, stop
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        model_dir='model.pt'
        torch.save(my_model.state_dict(), model_dir)
        valid_loss_min = valid_loss
        no_improvement=0
    else:
        no_improvement+=1
    if patience<no_improvement:
        print("Early stopping")
        break


dict = {'train_acc': train_acc_list, 'train_loss': train_loss_list, 'valid_acc': valid_acc_list, 'valid_loss':valid_loss_list}
df = pd.DataFrame(dict)

#Saving the dataframe
df.to_csv('model.csv')

#Evaluate the trained model
my_model.eval()
eval.evaluate(test_loader,model_dir)
