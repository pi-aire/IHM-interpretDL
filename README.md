# TP IHMxIA (non noté): interpretable Deep Reinforcement Learning
BRUNEAU Richard - VASLIN Pierre

### Saliency Maps
#### 1.
Voir la vidéo ![vidéo-1](https://github.com/pi-aire/IHM-interpretDL/blob/main/src/video/videoQ1_1.mp4)

#### 2.
Voir la vidéo ![vidéo-2](https://github.com/pi-aire/IHM-interpretDL/blob/main/src/video/videoQ2_1.mp4)

### Projection Umap
#### 1.
Pour récupérer la dernière couche convolutionnelle sous forme de vecteur, nous avons modifier le fichier [model.py](https://github.com/pi-aire/IHM-interpretDL/blob/main/src/model.py) plus précisément la méthode `forward`. Puis pour récupérer à chaque décision le vecteur, on a modifier le fichier [agent.py](https://github.com/pi-aire/IHM-interpretDL/blob/main/src/agent.py) pour qu'il puisse stocker dans un attribut `vecteurs` le vecteur produit à chaque décision d'action prise par l'agent. Dans [run.py](https://github.com/pi-aire/IHM-interpretDL/blob/main/src/run.py) nous récupérons via l'agent les vecteurs produits pendant les N époques, on sauvegarde également l'action qui correspond au vecteur pour après pourvoir concevoir l'Umap qui sera exportée dans un csv.

#### 2. et 3.
Nous avons réalisé l'Umpa sur 1000 époques et nous avons colorisé les points par action via le fichier [projection.html](https://github.com/pi-aire/IHM-interpretDL/blob/main/src/projection.html).
