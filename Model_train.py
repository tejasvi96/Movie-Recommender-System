import numpy as np
import pandas as pd
import torch
from loguru import logger


logger.add("log_prec.log")

movie_file="moviemapping_imdb.txt"

#optional use to take intersection of the movies
links_file='links.csv'

#Actual user rating files This has to be pivotized
ratings_file="ratings.csv"

#User embedding features size
features = 200

embeddings_file='embeddings_imdb.pt'
#if use embedding then hybrid system otherwise standard VAE
use_embedding=1

# Takes two values masked_recall and recall
eval_metric='recall'

load_model=1

model_file='hybrid_imdb.pt'

logger.info(movie_file)
logger.info(links_file)
logger.info(ratings_file)
logger.info(embeddings_file)

# if use_embedding
# then dont use vecs but use inp to feed in model
# Use vecs as long and inp as float
#number of epochs
epochs=0

# The actual learning rate
lr = 0.0001

batch_size=500

# To calculate the score Recall @R matches top R movies
recall_top=20
if torch.cuda.is_available:
    device=torch.device("cuda")
else:
    device=torch.device("cpu")
with open (movie_file,'r') as fp:
    data=fp.readlines()

# image imdb index to key location mapping
# Remember the padding vector was added in the beginning as zero to represent a movie which the user does not watch
dict={}
for i in data:
    i=i.replace("\n","")
    k,v=i.split('\t')
    dict[int(v)]=int(k)+1

# Maybe try to get the embeddings of the 18 genres and match against author
#skip to here

# import pandas as pd
links=pd.read_csv(links_file,encoding="utf-8")

# import pandas as pd
df=pd.read_csv(ratings_file,encoding="utf-8")

del df['timestamp']

filename="modified_links_4.csv"
# import pandas as pd
df2=pd.read_csv(filename,encoding="utf-8")

ind = df.movieId.isin(dict.keys())
df=df[ind]
df['rating']=[1 if i>=3.5 else 0 for i in df['rating'] ]

# Copying the movieid mapping from the dictionary so as to map to the embedding indexes
df['midd']=[dict[i] for i in df['movieId'] ]

# To actually perform the pivot opration. The normal pivot operation could not be used because of pandas limitations
# creates the sparse matrix representation
def create_matrix(data, user_col, item_col, rating_col):
    """
    creates the sparse user-item interaction matrix

    Parameters
    ----------
    data : DataFrame
        implicit rating data

    user_col : str
        user column name

    item_col : str
        item column name

    ratings_col : str
        implicit rating column name
    """



    # create a sparse matrix of using the (rating, (rows, cols)) format
    data[user_col]=data[user_col].astype('category')
    data[item_col]=data[item_col].astype('category')
    rows = data[user_col].cat.codes
    cols = data[item_col].cat.codes
    rating = data[rating_col]
    ratings = csr_matrix((rating, (rows, cols)),dtype=np.int16)
    return ratings, data


from scipy.sparse import csr_matrix
df2=create_matrix(df,'userId','midd','rating')
temp=df2[0].toarray()
user_movie_data=np.array(temp,dtype=np.int8)
temp=np.zeros(user_movie_data.shape[0],dtype=np.int8).reshape(-1,1)

print("Loaded DATA")
#This extra vector needs to be concatenated to make the indexes match
# actual usermovie data
t1=np.concatenate([temp,user_movie_data],axis=1)
# Pushing back to the original variable
user_movie_data=t1



mu_result=torch.load(embeddings_file)
vocab_size=len(dict)+1
inp_size=user_movie_data[0].shape[0]


# if random


if use_embedding==1:
    factor=3
else:
    factor=1

# define a simple linear VAE
import torch.nn as nn
class HybridVAE(nn.Module):
    def __init__(self):
        super(HybridVAE, self).__init__()
 
        # encoder
        if use_embedding:
            self.emb=torch.nn.Embedding(num_embeddings=vocab_size,embedding_dim=3).from_pretrained(mu_result)
        self.enc1 = nn.Linear(in_features=(inp_size)*factor, out_features=600)
        self.enc2 = nn.Linear(in_features=600, out_features=features*2)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=features, out_features=600)
        self.dec2 = nn.Linear(in_features=600, out_features=inp_size)
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        # encoding
#         x=x.to(device)
        if use_embedding:
            x=self.emb(x)
        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, features)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = F.softmax(self.dec2(x))
        return reconstruction, mu, log_var
 

# Design a VAE model for this
# batch_size = 64

import torch
model=HybridVAE().to(device)
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
 

print("Model definition done")
print(model)
from tqdm import tqdm
temp=[]
lossValues=[]
def fit(model,trainloader):
    model.train()
    running_loss = 0.0
    s=len(trainloader)

    for ind,data in tqdm(enumerate(trainloader), total=int(len(trainloader)/trainloader.batch_size)):
        inp,vec=data
        vec=vec.to(torch.int64)
        inp=inp.to(device)
        vec=vec.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(vec)
        bce_loss = criterion(reconstruction, inp.float())
        loss = final_loss(bce_loss, mu, logvar)
#         print(loss.item())
        lossValues.append(loss.item())
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(trainloader.dataset)
    return train_loss
 


#Data preparation pipeline
import numpy as np
# user_movie_data=np.array(df2[0].todense())
users=user_movie_data.shape[0]
train_user_movie_data=user_movie_data[:int(0.8*users)]
test_user_movie_data=user_movie_data[int(0.8*users):int(0.9*users)]
val_user_movie_data=user_movie_data[int(0.9*users):]
train_user_cols_data=np.zeros_like(train_user_movie_data,np.int16)
n,m=train_user_cols_data.shape
for i in range(n):
    inds=(np.nonzero(train_user_movie_data[i]))
#     print(inds)
    train_user_cols_data[i][inds]=[inds]

val_user_cols_data=np.zeros_like(val_user_movie_data,np.int16)
n,m=val_user_cols_data.shape
for i in range(n):
    inds=(np.nonzero(val_user_movie_data[i]))
#     print(inds)
    val_user_cols_data[i][inds]=[inds]
    
# Preparing test data
# Masking 20% of the movies already watched
test_user_cols_data=np.zeros_like(test_user_movie_data,np.int16)
n,m=test_user_cols_data.shape
test_user_masked_movies=np.zeros_like(test_user_movie_data,np.int16)
for i in range(n):
    inds=(np.nonzero(test_user_movie_data[i]))
    s=np.random.permutation(inds[0].shape[0])
    inds=inds[0][s]
    inds1=inds[:int(inds.shape[0]*0.8)]
    inds2=inds[int(inds.shape[0]*0.8):]
    test_user_masked_movies[i][inds2]=1
    test_user_movie_data[i][inds2]=0
    test_user_cols_data[i][inds1]=[inds1]

from torch.utils.data import TensorDataset,DataLoader  
TrainDataset=TensorDataset(torch.from_numpy(train_user_movie_data),torch.from_numpy(train_user_cols_data))
trainloader=DataLoader(TrainDataset,batch_size=batch_size)
ValDataset=TensorDataset(torch.from_numpy(val_user_movie_data),torch.from_numpy(val_user_cols_data))
valloader=DataLoader(ValDataset,batch_size=batch_size)
TestDataset=TensorDataset(torch.from_numpy(test_user_movie_data),torch.from_numpy(test_user_cols_data),torch.from_numpy(test_user_masked_movies))
testloader=DataLoader(TestDataset,batch_size=batch_size)


# Validation function
def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_user_movie_data)/dataloader.batch_size)):
            inp,vec=data
            vec=vec.to(torch.int64)
            inp=inp.to(device)
            vec=vec.to(device)
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(vec)
            bce_loss = criterion(reconstruction, inp.float())
            loss = final_loss(bce_loss, mu, logvar)
    #         print(loss.item())
            lossValues.append(loss.item())
            running_loss += loss.item()
#             loss.backward()
    val_loss = running_loss/len(dataloader.dataset)
    return val_loss

# Training Logic
import torch.nn.functional as F
train_loss = []
val_loss = []
min_val_loss=1000
if load_model==0:
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss = fit(model,trainloader)
        val_epoch_loss = validate(model, valloader)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}")
        if val_epoch_loss<min_val_loss:
            min_val_loss=val_epoch_loss
            torch.save(model,"hybrid_imdb.pt")

    logger.info(train_loss)
    logger.info(val_loss)
else:
    model=torch.load(model_file)
    logger.log("model loaded successfully")

# model=torch.load('simple.pt')
def test_rec(model, dataloader,r):
    print("using normal precision")
    model.eval()
    prec_values=[]
    running_loss = 0.0
    dataloader=DataLoader(TestDataset,batch_size=1)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader)/dataloader.batch_size)):
            inp,vec,_=data
            vec=vec.to(torch.int64)
            inp=inp.to(device)
            vec=vec.to(device)
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(vec)
            inds=reconstruction.topk(1000).indices
            m=inp.shape[0]
            movs_values=[]
            inds=inds.detach()
            for l in range(m):
                b=np.nonzero(inp[l].detach())
                b=[k[0].item() for k in b]
                s=inds[l][:r]
                a=[1 for v in s if v in b]
                prec_values.append((sum(a)/r))
        bce_loss = criterion(reconstruction, inp.float())
        loss = final_loss(bce_loss, mu, logvar)
        lossValues.append(loss.item())
        running_loss += loss.item()
    val_loss = running_loss/len(dataloader.dataset)
    return val_loss,prec_values

# Testing logic which gives the output as the recall score
def test(model, dataloader,r):
    print("using masked recall")
    model.eval()
    recall_values=[]
    running_loss = 0.0
    dataloader=DataLoader(TestDataset,batch_size=1)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader)/dataloader.batch_size)):
            inp,vec,movs=data
            vec=vec.to(torch.int64)
            inp=inp.to(device)
            vec=vec.to(device)
            movs=movs.to(device)
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(vec)
            inds=reconstruction.topk(1000).indices
            m=inp.shape[0]
            movs_values=[]
            movs=movs.detach()
            for j in range(m):
                t=np.nonzero(inp[j].detach())
                t=[l[0].item() for l in t]
#                 print(t)
                l=0
                s=[]
                k=0
                temp=[l.item() for l in inds[j].detach()]
                s=[ind for ind in temp if ind not in t][:r]
                movs_values.append(s)
        
            for l in range(m):

                b=np.nonzero(movs[l].detach())
                b=[k[0].item() for k in b]
                a=[1 for v in movs_values[l] if v in b]
                recall_values.append((sum(a)/r))
            
            bce_loss = criterion(reconstruction, inp.float())
            loss = final_loss(bce_loss, mu, logvar)
    
            lossValues.append(loss.item())
            running_loss += loss.item()
    val_loss = running_loss/len(dataloader.dataset)
    return val_loss,recall_values

if eval_metric=='masked_recall':
    l,recall_values=test(model,testloader,r=recall_top)
else:
    l,recall_values=test_rec(model,testloader,r=recall_top)
print(l)
print(sum(recall_values)/len(recall_values))
logger.info(l)
logger.info(sum(recall_values)/len(recall_values))