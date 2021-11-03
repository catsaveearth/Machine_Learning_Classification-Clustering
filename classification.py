import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import preprocessing, metrics
from sklearn.model_selection import GridSearchCV


dt_entropy_best_combi = {}
svm_best_combi = {}

gaussian_best_acc = {}
dt_entropy_best_acc = {}
svm_best_acc = {}


#for one-hot-encoding
def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix = column)], axis=1)
        data = data.drop(column, axis=1)
    return data

def Preprocessing(feature, encode_list, scale_list):
    # feature : dataframe of feature

    #scaler
    scaler_stndard = preprocessing.StandardScaler()
    scaler_MM = preprocessing.MinMaxScaler()
    scaler_robust = preprocessing.RobustScaler()
    scaler_maxabs = preprocessing.MaxAbsScaler()
    scaler_normalize = preprocessing.Normalizer()
    scalers = [None, scaler_stndard, scaler_MM, scaler_robust, scaler_maxabs, scaler_normalize]    
    scalers_name = ["original", "standard", "minmax", "robust", "maxabs", "normalize"]

    # encoder
    encoder_ordinal = preprocessing.OrdinalEncoder()
    #one hot encoding => using pd.get_dummies() (not used preprocessing.OneHotEncoder())
    encoders_name = ["ordinal", "onehot"]

    # result box
    result_dictionary = {}
    i = 0

    if encode_list == []:
        for scaler in scalers:
            if i == 0: #not scaling
                result_dictionary[scalers_name[i]] = feature.copy()

            else:
                #===== scalers
                result_dictionary[scalers_name[i]] = feature.copy()
                result_dictionary[scalers_name[i]][scale_list] = scaler.fit_transform(feature[scale_list]) #scaling
            i = i + 1
        return result_dictionary


    for scaler in scalers:
        if i == 0: #not scaling
            result_dictionary[scalers_name[i] + "_ordinal"] = feature.copy()
            result_dictionary[scalers_name[i] + "_ordinal"][encode_list] = encoder_ordinal.fit_transform(feature[encode_list])
            result_dictionary[scalers_name[i] + "_onehot"] = feature.copy()
            result_dictionary[scalers_name[i] + "_onehot"] = dummy_data(result_dictionary[scalers_name[i] + "_onehot"], encode_list)

        else:
            #===== scalers + ordinal encoding
            result_dictionary[scalers_name[i] + "_ordinal"] = feature.copy()
            result_dictionary[scalers_name[i] + "_ordinal"][scale_list] = scaler.fit_transform(feature[scale_list]) #scaling
            result_dictionary[scalers_name[i] + "_ordinal"][encode_list] = encoder_ordinal.fit_transform(feature[encode_list]) #encoding

            #===== scalers + OneHot encoding
            result_dictionary[scalers_name[i] + "_onehot"] = feature.copy()
            result_dictionary[scalers_name[i] + "_onehot"][scale_list] = scaler.fit_transform(feature[scale_list]) #scaling
            result_dictionary[scalers_name[i] + "_onehot"] = dummy_data(result_dictionary[scalers_name[i] + "_onehot"], encode_list) #encoding

        i = i + 1

    return result_dictionary


# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# pre-processing
print("=== 1. Data Load & Missing Data check")
print(train.describe())
print(test.describe())

# drop non-used feature
train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
test = test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# make test label from gender_submission.csv
test_label_list = pd.read_csv("gender_submission.csv")
test_label_list = test_label_list.sort_values(by='PassengerId', ascending=True)

for i in range(0, len(test)):
    passID = test.loc[i, 'PassengerId']

    for j in range(0, len(test_label_list)):
        comppassID = test_label_list.loc[j, 'PassengerId']

        if passID == comppassID:
            test.loc[i, 'Survived'] = test_label_list.loc[j, 'Survived']
            break


# check for missing values
print(train.isna().sum())
print("\n")
print(test.isna().sum())

# fill missing values
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

# check for missing values in training data (after filling then)
print("=== 2. fill missing data")
print(train.isna().sum())
print("\n")
print(test.isna().sum())


print("=== 3. drop not use data (unique ID)")
train_label = train.loc[:, "Survived"]
train = train.drop(['Survived'], axis=1)
test_label = test.loc[:, "Survived"]
test = test.drop(['PassengerId', 'Survived'], axis=1)
print(train.head())
print(test.head())


# scaling & encoding
print("=== 5. scaling & encoding")
train = Preprocessing(train, ["Sex"], ["Pclass", "Age", "SibSp", "Parch", "Fare"])
test = Preprocessing(test, ["Sex"], ["Pclass", "Age", "SibSp", "Parch", "Fare"])


print("=== 6. make model & testing")
K_list = [2, 5, 8, 10]
dt_param_list = {'max_depth': np.arange(3, 15), 'max_features' : ["sqrt", "log2"]}
svm_param_list = {"kernel" : ["linear", "poly", "rbf", "sigmoid"]}

for key, value in train.items():
    test_set = test[key] #get same scaling & encoding test data

    gaussian_model = GaussianNB()
    dt =  DecisionTreeClassifier(criterion="entropy") # decision Tree (entropy)
    svm = SVC() # support vector meachine

    dt_gini_param = None
    svm_param= None

    gaussian_acc = 0
    dt_entropy_acc = 0
    svm_acc = 0

    # gaussian classifier (naive bayes)
    gaussian_model.fit(value, train_label)
    gaussian_predicted = gaussian_model.score(test_set, test_label) 
    gaussian_acc = gaussian_predicted

    for k in K_list:
        # DT (entropy)
        dt_model = GridSearchCV(dt, dt_param_list, cv = k)
        dt_model.fit(value, train_label)
        dtEnt_bestModel = dt_model.best_estimator_
        dtEnt_score = dtEnt_bestModel.score(test_set, test_label)

        if dt_entropy_acc < dtEnt_score:
            dt_entropy_acc = dtEnt_score
            dt_model.best_params_["K"] = k
            dt_entropy_param = dt_model.best_params_

        # SVM
        svm_model = GridSearchCV(svm, svm_param_list, cv = k)
        svm_model.fit(value, train_label)
        svm_bestModel = svm_model.best_estimator_
        svm_score = svm_bestModel.score(test_set, test_label)

        if svm_acc < svm_score:
            svm_acc = svm_score
            svm_model.best_params_["K"] = k
            svm_param = svm_model.best_params_

    dt_entropy_best_combi[key] = dt_entropy_param
    svm_best_combi[key] = svm_param

    gaussian_best_acc[key] = gaussian_acc
    dt_entropy_best_acc[key] = dt_entropy_acc
    svm_best_acc[key] = svm_acc


print("=== 7. Result")
# get max accuray for each models
gaussian_bestcase = max(gaussian_best_acc)
dt_ent_bestcase = max(dt_entropy_best_acc, key=dt_entropy_best_acc.get)
svm_bestcase = max(svm_best_acc, key=svm_best_acc.get)

print("=> best acc")
print("Gaussian best acc : ", gaussian_best_acc[gaussian_bestcase])
print("Decision tree (entropy) best acc : ", dt_entropy_best_acc[dt_ent_bestcase])
print("support vector meachine (entropy) best acc : ", svm_best_acc[svm_bestcase])

print("\n=> best param")
print("Gaussian best param : ", gaussian_bestcase)
print("Decision tree (entropy) best param : ", dt_ent_bestcase, dt_entropy_best_combi[dt_ent_bestcase])
print("support vector meachine (entropy) best param : ", svm_bestcase, svm_best_combi[svm_bestcase])