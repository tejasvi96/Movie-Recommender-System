# Movie Recommender System using Collaborative Filtering

# Background

In this project we consider the problem of recommender system where users interact with the set of items and the aim of the businesses is to maximize the user engagement. Collaborative filtering is one such approach where suggestions based on the other similar users can be generated. We considered VAE's for this problem which help us to model a user using a distribution over the latent vectors to represent the users.  [here](https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_45.pdf)

Traditional approaches involve Matrix Factorization of the binarized matrix using linear methods. We make use of the Variational Autoencoders to arrive at a better representation by encoding each user over a distribution of latent feature vectors. 

Standard Autoencoder approach involves simply using a user*movie binarized vector but we introduce a modality in terms of movie features for each of the movie and use that to arrive at a more meaningful representation for a user.

To learn the movie embedding, two types of features set are explored-
1) Movie Genres where 18 genres are each used as a different feature.
2)Plot outlines of the movies are extracted from IMDB and sentiment analysis pipeline is used to produce a feature set for a movie which is used to produce the corresponding embedding.

![Images.](https://github.com/tejasvi96/Movie-Recommender-System/blob/main/Model.png?raw=True)

The evaluation methodology is for users in the test set we use Recall @R to match the top R model predictions for a user with the movies he/she has already watched


# Experiment

This problem makes use of the popular MovieLens 20M dataset  which can be downloaded from [here](https://grouplens.org/datasets/movielens/20m/).
To make use of the above code- 

For data preprocessing, set the filenames in the file Feature_extract.py

```
import pandas as pd
from empath import Empath
lexicon = Empath()
filename="links.csv"
```
For generating the movie embeddings , set the attributes in the file Embeddings_train.py

```
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
```
To actually train the model, set the attributes in the Model_Train.py file


```
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
```

To make predictions using the already trained model, use the make_predictions.py file ans set the movie_regex_to_match parameter as well

```

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
```
# Results

![Images.](https://github.com/tejasvi96/Movie-Recommender-System/blob/main/Screenshot_2020-12-23%20JupyterLab.png?raw=True)
![Images.](https://github.com/tejasvi96/Movie-Recommender-System/blob/main/Screenshot_2020-12-23%20JupyterLab_1.png?raw=True)
![Images.](https://github.com/tejasvi96/Movie-Recommender-System/blob/main/Screenshot_2020-12-28%20JupyterLab.png?raw=True)


