# Projet de El-bekri rajae ,Laaroussi lamyae et Khellouf leila: 
# Intitulé : clickbait_detection:
# Définition: 
- Clickbaiting est une stratégie d’utilisation intensionnelle pour les liens, tweets ou publications trompeuses sur les réseaux sociaux pour attirer en ligne les téléspectateurs.
# Exemples d’un clickbaits: 
- “The Hot New Phone Everybody Is Talking About”
- “You ll Never Believe Who Tripped and Fell on the Red Carpet”

Il existe plusieurs méthodes pour détecter automatiquement les clickbaits.
nous nous sommes basées sur l'article suivant pour construire nos différents modèles __(https://arxiv.org/abs/1612.01340)__

# BiDiractional RNN architecture for detecting Clickbait
![alt tag](https://user-images.githubusercontent.com/58788146/70670339-00e74b00-1c79-11ea-9cbd-3e22fb9551e7.png)
# Evaluation: 
_ Dataset= 15000 news headlines (7500 de clickbait source BuzzFeed , Upworthy et 7500 non_clickbait source Wikinews)
_ 10 fold cross validation
_ pour le training ils ont utilisés mini_batch gradient descent avec: 
_ batch_size=64
_ optimizer= ADAM
_ loss function = Binary Cross Entropy
_ Dropout =0.3
# Features: 
-  word embeddings  "word2vec embeddings de Google News dataset" ( afin de capturer les caractéristiques lexicales et sémantiques). 
-  character embeddings avec 3 couches de 1 dimension de CNN avec comme fonction d'activation ReLU et un Max_pooling .(afin de capturer les caractéristiques orthographiques et morphologiques).
# model type :
- Afin de capturer les informations contextuelles Il ont explorés différentes architectures du réseau de neurones récurrent comme: 
* Long Short Term Memory (LSTM)
* Gated Recurrent Unit (GRU) 
* standard RNNs
# Amélioration Approtée:
* Features: Utilisation de caractere embeddings ou lieu de Word Embeddings  
* Nouvelle Dataset de 32000 new headlines 
* Evalué avec un batch_size=128_ epochs=20 _ optimizer= ADAM _ loss function = Binary Cross Entropy et un Dropout =0.3


