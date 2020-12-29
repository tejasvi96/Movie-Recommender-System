# Code to predict the movies

import pandas as pd
import torch
import numpy as np
movie_file='moviemapping_imdb.txt'

model_file='hybrid_imdb.pt'

ratings_file='ratings.csv'
with open (movie_file,'r') as fp:
    data=fp.readlines()

# Set the input user movies here
movie_regex_to_match='terminator'
# image imdb index to key location mapping
# Remember the padding vector was added in the beginning as zero to represent a movie which the user does not watch
dict={}
for i in data:
    i=i.replace("\n","")
    k,v=i.split('\t')
    dict[int(v)]=int(k)+1

# Maybe try to get the embeddings of the 18 genres and match against author
#skip to here
links_file='links.csv'
# import pandas as pd
links=pd.read_csv(links_file,encoding="utf-8")

# import pandas as pd
df=pd.read_csv(ratings_file,encoding="utf-8")

del df['timestamp']
ind = df.movieId.isin(dict.keys())
df=df[ind]
df['rating']=[1 if i>=3.5 else 0 for i in df['rating'] ]

# Copying the movieid mapping from the dictionary so as to map to the embedding indexes
df['midd']=[dict[i] for i in df['movieId'] ]

data_movies=pd.read_csv("movies.csv")
data_movies=pd.merge(data_movies,df,on='movieId')
df4=data_movies[['title','midd']]
df4=df4.drop_duplicates()
inp_size=len(dict)

use_embedding=1
factor=3
features=200
device=torch.device("cuda")
lr=0.0001
import torch.nn as nn
import torch.nn.functional as F
vocab_size=len(dict)+1
class HybridVAE(nn.Module):
    def __init__(self):
        super(HybridVAE, self).__init__()
 
        # encoder
        if use_embedding:
            self.emb=torch.nn.Embedding(num_embeddings=vocab_size,embedding_dim=3)
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
#          data=data.squeeze(dim=1)
#         print(data.shape)
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


model=torch.load(model_file)

def make_input(mids):
    inp=torch.zeros(inp_size).reshape(1,-1)
    n=inp.shape[1]
    for i in range(n):
        if i in mids:
            inp[0][i]=1
    inp_vec=torch.zeros(inp_size,dtype=torch.long).reshape(1,-1)
    for i in range(n):
        inds=(np.nonzero(inp[0]))
        inp_vec[0][inds]=inds
    return inp,inp_vec

def make_prediction(inp,inp_vec):
    with torch.no_grad():
        inp,vec=inp,inp_vec
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
    return list(s.cpu().numpy())

print("I have watched")
print(df4[df4.title.str.contains(movie_regex_to_match)]['title'].values)
ids=df4[df4.title.str.contains(movie_regex_to_match)]['midd'].astype(int)
inp,vec=make_input(ids)
recommended_ids=make_prediction(inp,vec)
print("Model Recommends to me")
for i in recommended_ids:
    print(df4[df4['midd']==i]['title'].values)