# Machine_Learning_Classification-Clustering

Machine Learning Programming Homework 1~2, Department of Software, Gachon Univ, South Korea. (2021 fall semester)


## Preprocessing
### *Preprocessing(dataset, encode_list, scale_list)* <br>
: Apply data preprocessing(scaling & encoding) to dataset <br> <br>

**parameters:** <br>
    dataset : dataframe.  <br>
    encode_list : list to encode feature <br>
    scale_list : list to scale feature <br>
    
**return:** <br>
    dictionary to dataframe<br><br>
    
**Examples**
```
//classification
train = Preprocessing(train, ["Sex"], ["Pclass", "Age", "SibSp", "Parch", "Fare"])

//clustering
pre_feature = Preprocessing(dataset, ["ocean_proximity"], ["longitude", "latitude", "housing_median_age", "total_rooms", "population", "households", "median_income"])
```

reference : https://github.com/catsaveearth/scale_encode_combination


## Classification
Data load -> data preprocessing -> model training -> check result

**model:** <br>
1. DecisionTreeClassifier (entropy) <br>
2. Support vector machine (SVC) <br>
3. GaussianNB <br> <br>


## Clustering
Data load -> data preprocessing -> model training -> check result

**model:** <br>
1. k-mean <br>
2. EM (GaussianMixture) <br>
3. Clarans <br>
4. DBSCAN <br>
5. Meanshift <br>


