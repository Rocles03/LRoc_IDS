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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


#recuperation des donnees
dst_web = pd.read_csv('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')

#decoupage du jeu de donnees
X=dst_web.iloc[:,[5 ,63 , 66, 67]].values 
Y=dst_web.iloc[:,-1].values 
labEncr_Y = LabelEncoder()
Y = labEncr_Y.fit_transform(Y)
#verifiaction des donnees nulles 
print(dst_web.isnull().sum)

#fractionnenet du jeu de donnees
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

#Normalisation de mon jeu de donnees ou Feature Scaling
stand = StandardScaler()
X_train =stand.fit_transform(X_train)
X_test =stand.fit_transform(X_test)
#Creation du Classificateur de l'abre de decision 
#Importons le RandomForestClassifier de Scikit-learn
from sklearn.tree import DecisionTreeClassifier

#Initialisation d'un classifieur decision Tree avec des parametres par defaut
Dectree = DecisionTreeClassifier(criterion = 'gini',  random_state=50)
#adaptons le modele a nos donnees d'entrainement
Dectree.fit(X_train,Y_train)

#Evaluation de son exactitude a partir des donnees de test   
test_score =Dectree.score(X_test,Y_test)
print("Test score: %.2f%%" %(test_score*100.0) )

Prediction =Dectree.predict(X_test)

#Evaluation a l'aide la matrice de confusion 
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test, Prediction)

# Optimisationdu modele 

#initialisation du dictionnaire de valeurs d;hyperparametre
grid_params ={
    
        'max_depth':[1, 2, 3, 4, 5 ,6],
        'min_samples_leaf':[0.02, 0.04, 0.06 , 0.06]
        }

grid_objects =GridSearchCV(estimator =Dectree , param_grid=grid_params , scoring ='accuracy' , cv =10)

#Ajustons ensuite cet objet de grille aux donnees dapprentissage 
grid_objects.fit(X_train, Y_train)
import paralleleTestModule 
import multiprocessing as mp

'''if __name__ == '__main__':
    extractor = paralleleTestModule.ParallelExtractor()
    extractor.runInParallelel(numProcesses=2 ,numThreads=4)
    rf_random=RandomizedSearchCV(estimator = Random_FrsCls, param_distributions=grid_params ,n_iter = 1 , cv = 3 , verbose= 2 ,random_state = 42 ,n_jobs= -1 )
    rf_random.fit(X_train, Y_train)
    
    rf_random.best_params_'''

























