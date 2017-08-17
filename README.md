===================

Vous trouverez ici les explications sur comment utiliser les différents scripts fournis et quelques rapides explications sur leur fonctionnement.

#### Prérequis

Pour pouvoir utiliser les différents scripts et algorithmes, voici ce qu'il faut sur votre machine :

* Python2.7 ou Python3
* Divers modules python :
	* matplotlib :
	` $ sudo apt-get install python-matplotlib`
	* pandas :
	`$ sudo apt-get install python-pandas`
	* sklearn :
	 `$ sudo apt-get install python-sklearn`
	* numpy :
	`$ sudo apt-get install python-numpy`
* Des fichiers d'entraînement et de test au format csv.

#### Lancer un script

Pour lancer un des script, il suffit de lancer la commande suivante dans un terminal :
> $ python 'nom_du_script'

Il est possible de changer les fichiers d'entraînements et de tests, pour cela il faut utiliser trois arguments :
> - argument 1 : correspond aux données d'entraînements
> - argument 2 : correspond aux résultats des classes auquelles appartiennent les données d'entrainements
> - argument 3 : correspond aux données de tests

Si moins de trois arguments sont passés en paramètres, on prendra les données inclues.

#### Comprendre les différentes algorithmes

Vous avez ici la possibilité d'utiliser différents scripts et, par conséquent, différents algorithmes de random forest.
Voici une liste des différents algorithmes :

> - RandomForestClassifier
> - AdaBoostClassifier
> - ExtraTreesClassifier
	- DecisionTreeClassifier

#####  RandomForestClassifier

	L'algorithme de Random Forest est un algorithme de classification supervisé. Comme son nom l'indique, cet algorithme crée une forêt avec plusieurs arbres de décision. En général, dans une forêt, plus il y a d’arbres, plus celle-ci semble robuste. De la même manière, dans le classificateur de Random Forest, plus il y a d’arbres, plus l’algorithme est fiable.

##### AdaBoostClassifier

	Les différents classificateurs sont pondérés de manière à ce qu’à chaque prédiction, les classificateurs ayant prédit correctement auront un poids plus fort que ceux dont la prédiction était incorrecte.

#### ExtraTreesClassifier

	Cet algorithme met en oeuvre un meta-estimateur qui correspond à un certain nombre d’arbres de décision aléatoires sur différents sous-échantillons de l’ensemble de données et utilise la moyenne pour améliorer la précision prédictive et contrôler l’ajustement excessif.

#### DecisionTreeClassifier

	Un arbre de décision est une technique d’apprentissage supervisé. Comme son nom le suggère, c’est un outil d’aide. Il permet d’émettre des prédictions à partir des données connues sur le problème par réduction, niveau par niveau, du domaine des solutions, c’est-à-dire qu’il va déterminer la catégorie de chaque donnée à l’aide de règles de décision. L’arbre se construit petit à petit, à chaque fois qu’il va prendre une nouvelle décision de classification ; il va créer de nouvelles branches. Ainsi, à la fin on obtiendra un arbre qui à partir de variables en entrée pour prédire sa classe.

#### Enregistrer/charger le résultat d'un entraînement

	- Enregistrement :
	L'enregistrement d'un entraînement se fait grâce au module joblib de sklearn. Il s'utilise simplement en lancant la commande suivante dans le script :
	'joblib.dump(clf, 'model.pkl')'
	Avec clf l'objet correspondant au résultat de la commande fit (l'entraînement) et model.pkl le nom du fichier qui sera enregistré.

	- Chargement :
	Le chargement est tout aussi simple. Il suffit de lancer la commande suivante :
	'clf = joblib.load('model.pkl')'
	avec model.pkl le nom du fichier à charger, et clf qui sera l'entraînement chargé.
	L'objet obtenu peut ensuite être utilisé comme le résultat d'un entraînement normal.
----------
