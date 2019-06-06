import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, confusion_matrix
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from mlxtend.classifier import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras.layers import Dense
from keras.models import Model, Input
import keras.backend
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from src.create_grid_object import Grid
from copy import deepcopy

class ModelTrainer:
    def __init__(self,grid_obj: Grid,train_total=True):
        self.grid = deepcopy(grid_obj)
        self.train_total = train_total
        self.status = "Training classifier"
        print(self.status)

    def __getPerformance__(self,test_output,pred_output,classes):
        scr = f1_score(test_output, pred_output, average='macro')

        #cnf_mat = confusion_matrix(test_output,pred_output,labels=classes)
        cnf_mat = {}
        scr_array = f1_score(test_output,pred_output,labels=classes, average=None)

        for i in range(len(classes)):
            cnf_mat[classes[i]] = scr_array[i]



        # print("Accuracy score:",accuracy_score(test_output,pred_output))
        # print("f1 score:", f1_score(test_output, pred_output, average='weighted'))
        # print("report:", classification_report(test_output, pred_output))
        # print(confusion_matrix(test_output,pred_output))

        return scr, cnf_mat

    def __train_dnn__(self,train_input, test_input, train_output, test_output):

        # When using Tensorflow as backend, start with a fresh new session and avoid running out of memory
        keras.backend.clear_session()

        scaler = MinMaxScaler()
        scaler.fit(train_input)
        scaled_train_input = scaler.transform(train_input)
        scaled_test_input = scaler.transform(test_input)

        encoder = LabelEncoder()
        encoder.fit(train_output)
        encoded_train_output = encoder.transform(train_output)
        encoded_train_output = encoded_train_output.reshape(-1, 1)
        oh_enc = OneHotEncoder(sparse=False)
        oh_enc.fit(encoded_train_output)
        dummy_encoded_train_output = oh_enc.transform(encoded_train_output)

        num_layers = 5
        num_neurons = 100
        activation = 'relu'
        num_epochs = 1
        use_keras = False
        if use_keras:
            optimizer = 'Adam'


            inputs = Input(shape=(train_input.shape[1],))

            for i in range(num_layers):
                if i == 0:
                    hidden_layer = Dense(num_neurons, activation=activation)(inputs)
                else:
                    hidden_layer = Dense(num_neurons, activation=activation)(hidden_layer)

            outputs = Dense(dummy_encoded_train_output.shape[1], activation='softmax')(hidden_layer)

            classifier_model = Model(inputs=inputs, outputs=outputs)

            classifier_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            t_start = time.time()
            results = classifier_model.fit(scaled_train_input, dummy_encoded_train_output, verbose=0,
                                           epochs=num_epochs, validation_split=0.25)
            t_train = time.time() - t_start
            training_scr = results.history['val_acc']

        else:
            optimizer = 'adam'

            hidden_layers = [num_neurons]*num_layers
            classifier_model = MLPClassifier(hidden_layer_sizes=hidden_layers,activation=activation,solver=optimizer)

            t_start = time.time()
            results = classifier_model.fit(scaled_train_input, dummy_encoded_train_output)
            t_train = time.time() - t_start
            #training_scr = np.mean(cross_val_score(classifier_model, scaled_train_input, train_output, cv=4))
            training_scr = 0
        #print(t_train)
        t_start = time.time()
        dummy_encoded_pred_output = classifier_model.predict(scaled_test_input)
        t_pred = time.time() - t_start
        #print(t_pred)

        encoded_pred_output = np.argmax(dummy_encoded_pred_output, axis=1)
        pred_output = encoder.inverse_transform(encoded_pred_output)


        return classifier_model, pred_output

    def __train_rf__(self,train_input, test_input, train_output, test_output):
        rf = RandomForestClassifier()
        training_scr = np.mean(cross_val_score(rf,train_input,train_output,cv=4))
        rf.fit(train_input,train_output)
        pred_output = rf.predict(test_input)


        return rf, pred_output

    def __train_dt__(self,train_input, test_input, train_output, test_output):
        dt = DecisionTreeClassifier()
        #training_scr = np.mean(cross_val_score(dt,train_input,train_output,cv=4))
        training_scr = 0
        dt.fit(train_input,train_output)
        pred_output = dt.predict(test_input)

        return dt, pred_output

    def __train_gb__(self,train_input, test_input, train_output, test_output):
        gb = GradientBoostingClassifier()
        training_scr = np.mean(cross_val_score(gb,train_input,train_output,cv=4))
        gb.fit(train_input,train_output)
        pred_output = gb.predict(test_input)

        return gb, pred_output


    def __train_stacks__(self,train_input, test_input, train_output, test_output):
        scaler = MinMaxScaler()
        scaler.fit(train_input)
        scaled_train_input = scaler.transform(train_input)
        scaled_test_input = scaler.transform(test_input)
        clf1 = MLPClassifier(hidden_layer_sizes=[50,50,50,50],activation='logistic')
        clf2 = DecisionTreeClassifier()
        clf3 = SVC()
        clf4 = QuadraticDiscriminantAnalysis()
        clf5 = KNeighborsClassifier()
        clf6 = RandomForestClassifier()
        clf7 = LinearSVC()
        clf8 = LinearDiscriminantAnalysis()
        sclf1 = StackingClassifier(classifiers=[clf2, clf2, clf2, clf2],
                                  meta_classifier=clf6)
        sclf2 = StackingClassifier(classifiers=[clf7,clf7,clf7,clf7],
                                  meta_classifier=clf3)
        sclf3 = StackingClassifier(classifiers=[clf8,clf8,clf8,clf8],
                                   meta_classifier=clf4)
        sclf4 = StackingClassifier(classifiers=[clf5,clf5,clf5,clf5], meta_classifier=clf5)
        main_clf = StackingClassifier(classifiers=[sclf1,sclf2,sclf3,sclf4],meta_classifier=clf1)
        ada_main = AdaBoostClassifier(main_clf)
        training_scr = np.mean(cross_val_score(ada_main, scaled_train_input, train_output, cv=4))
        ada_main.fit(scaled_train_input, train_output)
        pred_output = ada_main.predict(scaled_test_input)

        return ada_main, pred_output


    def train(self,train_input, test_input, train_output, test_output,training_method="rf"):

        t_start = time.time()
        if training_method == "rf":
            clf, pred_output = self.__train_rf__(train_input, test_input, train_output, test_output)
        elif training_method == "dnn":
            clf, pred_output = self.__train_dnn__(train_input, test_input, train_output, test_output)
        elif training_method == "gb":
            clf, pred_output = self.__train_gb__(train_input, test_input, train_output, test_output)
        elif training_method == "dt":
            clf, pred_output = self.__train_dt__(train_input, test_input, train_output, test_output)
        else:
            clf, pred_output = self.__train_stacks__(train_input, test_input, train_output, test_output)

        scr = self.__getPerformance__(test_output, pred_output, clf.classes_)

        if self.train_total:
            self.grid.total_classification_accuracy = scr[0]
            self.grid.confusion_matrix_total = scr[1]
            self.grid.total_classifier_model = clf
            self.grid.total_classifier_model_training_time = time.time()-t_start
        else:
            self.grid.zone_classification_accuracy = scr[0]
            self.grid.confusion_matrix_zone = scr[1]
            self.grid.zone_classifier_model = clf
            self.grid.zone_classifier_model_training_time = time.time()-t_start

        self.status = "Classifier trained"
        print(self.status)

        return self.grid