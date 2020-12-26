import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import torch
import numpy as np

# ch=1 for IMDB 
# ch=2 fro 18 genres

ch=2
if ch==1:
    filename="modified_links_4.csv"
    df=pd.read_csv(filename,encoding="utf-8")
#     This should be on the basis of the feature type whether 1800 or 100
    df2=df[[i for i in df.columns if i not in ['movieId','imdbId','tmdbId','plot outline']]]
    data=torch.tensor(df2.values.astype(np.float32))
    inp_dim=85
    hidden_dim=50
if ch==2:
    inp_dim=18
    hidden_dim=10
    filename="movies.csv"
    import pandas as pd
    df=pd.read_csv(filename,encoding="utf-8")
    genres=set()
    for i in df.genres:
        old=0
        s=i
        while True:
            if(s.find("|",old)==-1):
                break
            new=s.find('|',old+1)
            genres.add(s[old:new])
            old=new+1
    import numpy as np
    one_hot=pd.get_dummies(list(genres))
    for i in list(genres):
        df[i]=0.0
    for index,trial in df.iterrows():
    #     print(i['title'])
        s=df.loc[index,'genres']
    #     print(s)
        old=0
        while True:
            if(s.find("|",old)==-1):
                df.loc[index,s[old:]]=1.0
                break
            new=s.find('|',old)
            df.loc[index,s[old:new]]=1.0
    #         print(s[old:new])
            old=new+1
    import numpy as np
    one_hot=pd.get_dummies(list(genres))
    for i in list(genres):
        df[i]=0.0
    for index,trial in df.iterrows():
    #     print(i['title'])
        s=df.loc[index,'genres']
    #     print(s)
        old=0
        while True:
            if(s.find("|",old)==-1):
                df.loc[index,s[old:]]=1.0
                break
            new=s.find('|',old)
            df.loc[index,s[old:new]]=1.0
    #         print(s[old:new])
            old=new+1
    df.dropna()
    df2=df[[i for i in df.columns if i not in ['movieId','title','genres','IMAX','(no genres listed)']]]
    data=torch.tensor(df2.values.astype(np.float32))
    
labels=torch.tensor(df['movieId'].values)
train_data=data[:int(0.8*(len(data)) )]
train_labels=labels[:int(0.8*(len(data)) )]
val_data=data[int(0.8*(len(data))) :]
val_labels=labels[int(0.8*(len(data))) :]
train_dataset=TensorDataset(train_data,train_labels)
val_dataset=TensorDataset(val_data,val_labels)
trainLoader=DataLoader(train_dataset,batch_size=50)
valLoader=DataLoader(val_dataset,batch_size=50)
features = 3
# define a simple linear VAE
import torch.nn as nn
class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Linear(in_features=inp_dim, out_features=hidden_dim)
        self.enc2 = nn.Linear(in_features=hidden_dim, out_features=features*2)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=features, out_features=hidden_dim)
        self.dec2 = nn.Linear(in_features=hidden_dim, out_features=inp_dim)
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
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, features)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction, mu, log_var


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
 
from tqdm import tqdm
def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_dataset)/dataloader.batch_size)):
        data, label = data
        data = data[0].to(device).reshape(1,-1)
#         print(data.shape)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss

def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            data, label = data
            data = data[0].to(device).reshape(1,-1)
            data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()

    val_loss = running_loss/len(dataloader.dataset)
    return val_loss


criterion = nn.BCELoss(reduction='sum')
batch_size = 64
lr = 0.0001
model=LinearVAE()
device=torch.device("cuda")
model=model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
import torch.nn.functional as F
train_loss = []
val_loss = []
epochs=50
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model, trainLoader)
    val_epoch_loss = validate(model, valLoader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")



torch.save(model,"genre_embeddings_model.pt")
mu_output = []
logvar_output = []
movie_ids=[]
# get the embeddings
import torch.nn.functional as F
with torch.no_grad():
    for i, (data) in enumerate(trainLoader):
            movie_ids.append(data[1].reshape(-1,1))
            data = data[0].to(device)
#             optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            mu_tensor = mu   
            mu_output.append(mu_tensor)
            mu_result = torch.cat(mu_output, dim=0)
            logvar_tensor = logvar   
            logvar_output.append(logvar_tensor)
            logvar_result = torch.cat(logvar_output, dim=0)
with torch.no_grad():
    for i, (data) in enumerate(valLoader):
            movie_ids.append(data[1].reshape(-1,1))
            data = data[0].to(device)
#             optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            mu_tensor = mu   
            mu_output.append(mu_tensor)
            mu_result = torch.cat(mu_output, dim=0)
            logvar_tensor = logvar   
            logvar_output.append(logvar_tensor)
            logvar_result = torch.cat(logvar_output, dim=0)
movies=[]
for i in movie_ids:
    for j in i:
        movies.append(j.item())

dummy=torch.zeros((3)).reshape(1,-1).to(device)
mu_result=torch.cat((dummy,mu_result),dim=0)

moviemap={}
for i in range(len(movies)):
    moviemap[i]=movies[i]

with open ("moviemapping_genre.txt",'w') as fp:
    for k in moviemap.keys():
        fp.write(str(k)+"\t"+str(moviemap[k])+"\n")

torch.save(mu_result,"embeddings_genre.pt")
mu_result=torch.load('embeddings_genre.pt')


# vocab_size=22549+1

import numpy as np
from sklearn.manifold import TSNE
# X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# X_embedded = TSNE(n_components=2).fit_transform(mu_result.cpu())
# X_embedded.shape


#todo change this for the movies
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=18, random_state=0).fit(mu_result.cpu())
# res1=kmeans.predict(mu_result.cpu())

# import seaborn as sns
# import matplotlib.pyplot as plt
# palette = sns.color_palette("dark", 18)
# sns.scatterplot(X_embedded[:,0], X_embedded[:,1],hue=res1, legend='full',palette=palette)
# plt.title("Movie clustering into 18 genres")

