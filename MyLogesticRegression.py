import numpy as np

class MyLogisticRegression:

    def __init__(self, weights, constants, labels):
        self.weights=weights
        self.constants=constants
        self.labels=labels



    def predict_proba(self,X):

        prob=[]
        X=X.toarray()


        for index,x in enumerate(X):
            numerator=[]
            i=0

            for weight in self.weights:
                numerator.append(np.exp(np.dot(x, weight)+self.constants[i]))
                i+=1
            denominator=sum(numerator)
            prob.append([])

            for nume in numerator:
                prob[index].append(nume/denominator)



        return np.asarray(prob)



    def predict(self,X):
        prob=[]
        X=X.toarray()
        #print X.shape,'x.shape'


        for index,x in enumerate(X):
            numerator=[]
            i=0

            for weight in self.weights:
                numerator.append(np.exp(np.dot(x, weight)+self.constants[i]))
                i+=1
            denominator=sum(numerator)
            best_prob=-9e99
            best_label=None



            for index2,nume in enumerate(numerator):
                class_prob=nume/denominator
                if class_prob > best_prob:
                    best_prob=class_prob
                    best_label=self.labels[index2]

            prob.append(best_label)





        return np.asarray(prob)