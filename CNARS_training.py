###############################################################################
# C-NARS: An Open-Source Tool for Classification of Narratives in Survey Data 
# (1) CNARS_main.py: Performs a classification task on unlabeled test data given labeled training data        #
# (2) CNARS_training.py: Performs and evaluates a classification task given labeled training data 
###############################################################################

# Authors    ################################################################## 
# Jinseok Kim (Ph.D., Survey Research Center & School of Information, University of Michigan Ann Abror)
# Jenna Kim (Doctoral Student, School of Information, University of Illinois at Urbana-Champaign)
# Joelle Abramowitz (Ph.D., Survey Research Center, University of Michigan)
###############################################################################

# Acknowledgement #############################################################
# Sponsor: Michigan Retirement and Disability Research Center (MRDRC)
# Grant Number: UM21-14 “What We Talk about When We Talk about Self-employment: Examining Selfemployment and the Transition to Retirement among Older Adults in the United States”
# PI: Joelle Abramowitz; co-PI: Jinseok Kim
# Project Period: 10/2020 ~ 09/2021
###############################################################################

###########################
# License: CC BY-NC 2.0 ###
###########################

import pandas as pd
import nltk
import re
import itertools

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix, classification_report

def load_data(filename):
    # Load input file
    df = pd.read_csv(filename, encoding='unicode_escape')

    print("Total No of Rows: ", df.shape[0])
    print("Total No of Columns: ", df.shape[1])
    
    print('\nTraining & Test Size(row, colum):')
    df.iloc[:, -1].value_counts()
  
    # Split data for training and test (Test size: 0.2, stratify turned on)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print('X_train: {}\nX_test: {}\ny_train: {}\ny_test: {}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
    
    return X_train, X_test, y_train, y_test

def sample_data(X_train, y_train, sampling=0, sample_method='over'):
    
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    
    if sampling:
        # select a sampling method
        if sample_method == 'over':
            oversample = RandomOverSampler(random_state=42)
            X_over, y_over = oversample.fit_resample(X_train, y_train)
            print('\nOversampled Data (class, Rows):\n{}'.format(y_over.value_counts()))
            X_train_sam, y_train_sam = X_over, y_over
            
        elif sample_method == 'under':
            undersample = RandomUnderSampler(random_state=42)
            X_under, y_under = undersample.fit_resample(X_train, y_train)
            print('\nUndersampled Data (class,Rows):\n{}'.format(y_under.value_counts()))
            X_train_sam, y_train_sam = X_under, y_under
    else:
        X_train_sam, y_train_sam = X_train, y_train      
        print('\nNo Sampling Performed\n')
    
    return X_train_sam, y_train_sam

def preprocess_data(X_data):
    
    # 1 convert all characters to lowercase
    X_data.iloc[:, 1] = X_data.iloc[:, 1].map(lambda x: x.lower())
     
    # 2. remove non-alphabetical characters
    X_data.iloc[:, 1] = X_data.iloc[:, 1].str.replace("[^a-zA-Z]", " ", regex=True)
    
    # remove stopwords in English
    stop_english = stopwords.words('english')
    X_data.iloc[:, 1] = X_data.iloc[:, 1].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_english)]))
    
    # remove stopwords in Spanish
    stop_spanish = stopwords.words('spanish')
    X_data.iloc[:, 1] = X_data.iloc[:, 1].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_spanish)]))
    
    # remove words
    words_to_remove = {'po', 'pi', 'ao', 'no'}
    X_data.iloc[:, 1] = X_data.iloc[:, 1].apply(lambda x: ' '.join([word for word in x.split() if word not in words_to_remove]))

    # remove words
    X_data.iloc[:, 1] = X_data.iloc[:, 1].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 1]))
        
    # 3. word tokenize
    X_data.iloc[:, 1] = X_data.iloc[:, 1].apply(nltk.word_tokenize)
    
    # 4. stemming
    stemmer = PorterStemmer()
    X_data.iloc[:, 1] = X_data.iloc[:, 1].apply(lambda x: [stemmer.stem(y) for y in x])
    
    # ngram
    #X_data.iloc[:, 1] = X_data.iloc[:, 1].apply(lambda x: list(nltk.ngrams(x, 2)))
    #X_data.iloc[:, 1] = X_data.iloc[:, 1].apply(lambda x: [''.join(i) for i in x])
    #print (X_data.iloc[0, 1])
    
    # 5. creating features using tf-idf
    X_data.iloc[:, 1] = X_data.iloc[:, 1].apply(lambda x: ' '.join(x))
       
    return X_data

def fit_model(X_train, y_train, model='DT'):
    
    if model=='DT':
        DT = DecisionTreeClassifier(max_depth=2)
        model = DT.fit(X_train, y_train)
    elif model=='SVM':
        SVM = SVC(kernel='linear', probability=True)  
        model = SVM.fit(X_train, y_train)
    elif model=='KNN':
        KNN = KNeighborsClassifier(n_neighbors=7)  
        model = KNN.fit(X_train, y_train)
    elif model=='NB':
        NB = MultinomialNB()
        model = NB.fit(X_train, y_train)
    elif model=='RF':
        RF = RandomForestClassifier(max_depth=2, random_state=0)
        model = RF.fit(X_train, y_train)
    elif model=='GB':
        GB = GradientBoostingClassifier()
        model = GB.fit(X_train, y_train)
    
    return model

def evaluate_model(y_test, y_pred, eval_model=0):
    
    if eval_model:
        print('\nConfusion Matrix:\n')
        print(confusion_matrix(y_test, y_pred))
    
        print('\nClassification Report:\n')
        print(classification_report(y_test, y_pred))

def predict_proba(model, X_test_trans, X_test, y_test, y_pred, proba_file):
    
    ## Compute probability
    y_prob = model.predict_proba(X_test_trans)
    df_prob = pd.DataFrame(data=y_prob, columns=model.classes_)
    result = pd.concat([X_test.reset_index(drop=True), df_prob], axis=1, ignore_index=False)

    ## Add predicted class to output
    result['pred'] = pd.Series(y_pred)

    ## Add actual class to output 
    y_test = y_test.reset_index(drop=True)
    result['act'] = y_test

    ## Save output
    result.to_csv(proba_file, encoding='utf-8-sig', index=False, header=True)

def main(input_file, sample_on, sample_type, model_method, eval_on, proba_file):
    
    ## 1. Load data
    X_train, X_test, y_train, y_test = load_data(input_file)
    
    print('\nOriginal Data (class, rows):\n{}'.format(y_train.value_counts()))
    
    ## 2. Sampling 
    X_train_samp, y_train_samp = sample_data(X_train, y_train, sampling=sampling_on, sample_method=sampling_type)
    
    ## 3. Preprocessing 
    X_train_pro = preprocess_data(X_train_samp)
    
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(X_train_pro.iloc[:, 1])
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True).fit(counts)
    X_train_transformed = transformer.transform(counts)
    
    X_train_trans = X_train_transformed
    y_train_trans = y_train_samp

    ## 4. Model Fitting
    model = fit_model(X_train_trans, y_train_trans, model=model_method)
    
    ## 5. Prediction
    # Transform X_test data
    X_test_pro = preprocess_data(X_test)
    counts_test = count_vect.transform(X_test_pro.iloc[:, 1])
    X_test_trans = transformer.transform(counts_test)
    
    # Predict output
    y_pred = model.predict(X_test_trans)
    
    ## 6. Evaluating model performance
    evaluate_model(y_test, y_pred, eval_model=eval_on)
    
    ## 7. Probability prediction    
    predict_proba(model, X_test_trans, X_test, y_test, y_pred, proba_file=proba_file)
    print("\nOutput file:'" + proba_file + "' Created")

if __name__== "__main__":
    
    ## Define parameter values

    input_file="data_training.csv"

    
    sampling_on=0             # 0 for no sampling; 1 for sampling
    sampling_type='over'      # Use when sampling_on=1; 'over'(oversampling), 'under'(undersampling)

    model_type='GB'           #'DT'(Decision Tree);'SVM'(SVM);'KNN'(KNeighbors);#'NB'(Naive Bayes);
                              #'RF'(Random Forest);'GB'(Gradient Boosting)
    
    eval_on=1                 # 0 for no; 1 for yes (display confusion matrix/classification report)
    
    output_file = "proba_" + model_type + "_train-test_.csv"  # Filename for probability output 
    
    
    ## Main fuction
    main(input_file=input_file, 
         sample_on=sampling_on, 
         sample_type=sampling_type, 
         model_method=model_type, 
         eval_on=eval_on, 
         proba_file=output_file)


