#imporattion des Lib
import statsmodels as stat
import seaborn as sbm
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import matplotlib as mplt
import numpy as np 
from sklearn.model_selection import train_test_split


#recuperation des donnees
dst_web = pd.read_csv('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
#description du jeu de donnees 
print(dst_web.describe())
#preparation des donnees
#visualisation des colonnes
print(dst_web.columns)
#decoupage du jeu de donnees
X=dst_web.iloc[:,[5 ,63 , 66, 67]].values 
Y=dst_web.iloc[:,-1].values 
labEncr_Y = LabelEncoder()
Y = labEncr_Y.fit_transform(Y)
#verifiaction des donnees nulles 
print(dst_web.isnull().sum)
#standardisation du jeu de donnees
stand = StandardScaler()
X_train =stand.fit_transform(X_train)
X_test =stand.fit_transform(X_test)

#fractionnenet du jeu de donnees
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
#Creation du Classificateur du random Forest
#Importons le RandomForestClassifier de Scikit-learn
from sklearn.ensemble import RandomForestClassifier

#Initialisation d'un classifieur de Foret aleatoire avec des parametres par defaut
Random_FrsCls = RandomForestClassifier(n_estimators =5000 , max_depth =1 ,min_samples_leaf =0.05 ,  random_state=50)
#adaptons le modele a nos donnees d'entrainement
Random_FrsCls.fit(X_train,Y_train)

#Evaluation de son exactitude a partir des donnees de test
test_score =Random_FrsCls.score(X_test,Y_test)
print("Test score: %.2f%%" %(test_score*100.0) )
