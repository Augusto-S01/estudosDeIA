from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



dataset_cancer = load_breast_cancer()
print(dataset_cancer.feature_names)
print(dataset_cancer.target_names)

dataset_diabetes = load_diabetes()
print (dataset_diabetes.feature_names)
print (dataset_diabetes.target)

#copilot

X_train_can, X_test_can, y_train_can, y_test_can = train_test_split(dataset_cancer.data, dataset_cancer.target, stratify=dataset_cancer.target,random_state=42)
# X_train_can e X_test_can é um conjunto de dados de treinamento e teste para o eixo X
# e no eixo Y também sera feito esse treinamento com y_train_can e y_test_can

#o método train_test_split recebe  o dataset_cancer como massa de testes sendo:
#dataset_cancer.data é o conjunto de variaveis que vao ser analisadas
#dataset_cancer.target são os resultados esperados

#o parametro stratify=é a classificação deles
#o random_state=42 é um comando para tentar pegar aleatoriamente oq é massa de dados de treino e o q massa de dados para teste

X_train_dia, X_test_dia, y_train_dia, y_test_dia = train_test_split(dataset_cancer.data, dataset_cancer.target, stratify=dataset_cancer.target, random_state=42)


#SVM

#acuracia de treino
training_accuracy = []
#acuracia de testes
test_accuracy = []

#fazendo o uso de tres kernel
#procurando se existe algo linear entre as informações
#uma relação polinomial
#ou uma sigmoidal
kernels = ['linear', 'rbf', 'sigmoid']

#para cada kernel sera feito um treino e teste
for kernel in kernels:

    #aqui ele executa o modelo
    svm_model = svm.SVC(kernel=kernel)
    #aqui ele faz o fit ( treino )
    svm_model.fit(X_train_can, y_train_can)
    #quarda a acuracia do treino
    training_accuracy.append(svm_model.score(X_train_can, y_train_can))
    #guarda a acuracia do teste
    test_accuracy.append(svm_model.score(X_test_can, y_test_can))


#aqui esse sequencia de comandos é para fazer o grafico com o matplot
plt.plot(kernels, training_accuracy, label='Acuracia no conj. treino')
plt.plot(kernels, test_accuracy, label='Acuracia no conj. teste')
plt.ylabel('Accuracy')
plt.xlabel('Kernels')
plt.legend()