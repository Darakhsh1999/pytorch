
def accuracy(y_target, y_prediction):
    # TODO assert dtype is integer
    return (y_target == y_prediction).sum() / len(y_target)

def precision(y_target, y_prediction):
    pass

def recall(y_target, y_prediction):
    pass

def F1(y_target, y_prediction):
    pass

def confusion_matrix(y_target, y_prediction):
    pass


# TODO remove this, we only want pytorch native tensors
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


A = np.array([0,1,1,2,3,3,4,4,4,4,4]) # target
B = np.array([0,1,2,2,3,4,1,2,3,3,4]) # prediction

# FP = we predict it is class i but it is not
# FN = we predict it is not class i but it actually is class i

accuracy = accuracy_score(A,B)
precision = precision_score(A,B, average="macro") # TP/(TP+FP)
recall = recall_score(A,B, average="macro") # TP/(TP+FN)
f1 = f1_score(A,B, average="macro") # 
conf_matrix = confusion_matrix(A,B, labels=np.arange(5)) # row = labels, columns = predictions
print(conf_matrix)

my_acc = np.diag(conf_matrix).sum() / conf_matrix.sum()
prec = []
for i in range(5):
    prec.append(conf_matrix[i,i]/conf_matrix[:,i].sum() if conf_matrix[:,i].sum() != 0 else 0)
prec = np.array(prec)
my_prec = prec.mean()

rec = []
for i in range(5):
    rec.append(conf_matrix[i,i]/conf_matrix[i,:].sum() if conf_matrix[i,:].sum() != 0 else 0)
rec = np.array(rec)
my_rec = rec.mean()

my_f1 = (2*(prec*rec)/(prec+rec)).mean()

print("-----")
print(accuracy)
print(my_acc)
print("-----")
print(precision)
print(my_prec)
print("-----")
print(recall)
print(my_rec)
print("-----")
print(f1)
print(my_f1)
print("-----")
