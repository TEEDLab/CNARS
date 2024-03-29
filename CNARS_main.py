###############################################################################
# C-NARS: An Open-Source Tool for Classification of Narratives in Survey Data 
# (1) CNARS_main.py: Performs a classification task on unlabeled test data given labeled training data        #
# (2) CNARS_training.py: Performs and evaluates a classification task given labeled training data 
###############################################################################

# Authors    ################################################################## 
# Jinseok Kim (Ph.D., Survey Research Center & School of Information, University of Michigan Ann Arbor)
# Jenna Kim (Doctoral Student, School of Information, University of Illinois at Urbana-Champaign)
# Joelle Abramowitz (Ph.D., Survey Research Center, University of Michigan)
###############################################################################

# Acknowledgement #############################################################
# Sponsor: Michigan Retirement and Disability Research Center (MRDRC)
# Grant Number & Title: UM21-14 “What We Talk about When We Talk about Self-employment: Examining Self-employment and the Transition to Retirement among Older Adults in the United States”
# PI: Joelle Abramowitz; co-PI: Jinseok Kim
# Project Period: 10/2020 ~ 09/2021
###############################################################################

###########################
# License: MIT          ###
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
    """
    Read in input file and load data
    
    filename: csv file
    
    """
    
    df = pd.read_csv(filename, encoding='unicode_escape')  #iso-8859-1

    print("No of Rows: ", df.shape[0])
    print("No of Columns: ", df.shape[1])
    
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    
    print('\nClass Counts(label, row):')
    print(y.value_counts())
    print('\n')
    
    return X, y

def sample_data(X_train, y_train, sampling=0, sample_method='over'):
    """
       Sampling input train data
       
       X_train: dataframe of X train data
       y_train: dataframe of y train data
       sampling: indicator of sampling funtion is on or off
       sample_method: method of sampling (oversampling or undersampling)
       
    """
    
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

def preprocess_data(X_data_raw):
    """
       Preprocess data with lowercase conversion, punctuation removal, tokenization, stemming, and tf-idf
       
       X_data_raw: X data in dataframe
       
    """
    
    X_data=X_data_raw.iloc[:, 1]
    
    # 1 convert all characters to lowercase
    X_data = X_data.map(lambda x: str(x).lower())
     
    # 2. remove non-alphabetical characters
    X_data = X_data.str.replace("[^a-zA-Z]", " ", regex=True)
    
    # remove stopwords in English
    stop_english = stopwords.words('english')
    X_data = X_data.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_english)]))
    
    # remove stopwords in Spanish
    stop_spanish = stopwords.words('spanish')
    X_data = X_data.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_spanish)]))
    
    # remove words 1: target
    words_to_remove = {'po', 'pi', 'ao', 'no'}
    X_data = X_data.apply(lambda x: ' '.join([word for word in x.split() if word not in words_to_remove]))

    # remove words 2: length
    X_data = X_data.apply(lambda x: ' '.join([word for word in x.split() if len(word) > 1]))
        
    # 3. word tokenize
    X_data = X_data.apply(nltk.word_tokenize)
    
    # 4. stemming
    stemmer = PorterStemmer()
    X_data = X_data.apply(lambda x: [stemmer.stem(y) for y in x])
    
    # ngram
    #X_data = X_data.apply(lambda x: list(nltk.ngrams(x, 2)))
    #X_data = X_data.apply(lambda x: [''.join(i) for i in x])
    #print (X_data[0, 1])
    
    # join by spaces 
    X_data = X_data.apply(lambda x: ' '.join(x))
       
    return X_data

def fit_model(X_train, y_train, model='DT'):
    
    """
      Model fitting with options of classifiers:
      decision tree, svm, knn, naive bayes, random forest, and gradient boosting
      
      X_train: X train data
      y_train: y train data
      model: name of classifier
      
    """
    
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
    """
      evaluate model performance
      
      y_test: y test data
      y_pred: t prediction score
      eval_model: indicator if this funtion is on or off
      
    """
    
    if eval_model:
        print('****** Model Evaluation ******')
        print('\nConfusion Matrix:\n')
        print(confusion_matrix(y_test, y_pred))
    
        print('\nClassification Report:\n')
        print(classification_report(y_test, y_pred))

def predict_proba(model, X_test_trans, X_test, y_test, y_pred, proba_file):
    """
       Predict probability of each class
       
       model: trained model with selected classifier
       X_test_trans: X test data preprocessed
       X_test: original X test data
       y_test: original y test data
       y_pred: predicted y values
       proba_file: output file of probability scores
       
    """
    
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

def main(input_train_file, input_test_file, sample_on, sample_type, model_method, eval_on, proba_file):
    
    """
       Main function for processing data, model fitting, and prediction
       
       input_train_file: input train file
       input_test_file: input test file
       sample_on: indicator of sampling on or off
       sample_type: sample type to choose if sample_on is 1
       model_method: name of classifier to be applied for model fitting
       eval_on: indicator of model evaluation on or off
       proba_file: name of output file of probability
       
    """
    
    ## 1. Load data
    
    print("****** Training Data ******")
    X_train, y_train = load_data(input_train_file)
    
    print("\n****** Test Data ******")
    X_test, y_test = load_data(input_test_file)
    
    ## 2. Sampling 
    X_train_samp, y_train_samp = sample_data(X_train, y_train, sampling=sampling_on, sample_method=sampling_type)
    
    ## 3. Preprocessing 
    X_train_pro = preprocess_data(X_train_samp)
    
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(X_train_pro)
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True).fit(counts)
    X_train_transformed = transformer.transform(counts)
    
    X_train_trans = X_train_transformed
    y_train_trans = y_train_samp

    ## 4. Model Fitting
    model = fit_model(X_train_trans, y_train_trans, model=model_method)
    
    ## 5. Prediction
    # Transform X_test data
    X_test_pro = preprocess_data(X_test)
    counts_test = count_vect.transform(X_test_pro)
    X_test_trans = transformer.transform(counts_test)
    
    # Predict output
    y_pred = model.predict(X_test_trans)
    
    ## 6. Evaluating model performance
    evaluate_model(y_test, y_pred, eval_model=eval_on)
    
    ## 7. Probability prediction    
    predict_proba(model, X_test_trans, X_test, y_test, y_pred, proba_file=proba_file)
    print("\nOutput file:'" + proba_file + "' Created")

if __name__== "__main__":
    
    ## 1. Set Parameter Values
    
    input_file_train="data_training.csv"  # input train dataset
    input_file_test="data_target.csv" #"data_test_selfemp.csv"   # input test dataset
      
    sampling_on=0             # 0 for no sampling; 1 for sampling
    sampling_type='over'      # Use when sampling_on=1; 'over'(oversampling), 'under'(undersampling)

    model_type='GB'           #'DT'(Decision Tree);'SVM'(SVM);'KNN'(KNeighbors);#'NB'(Naive Bayes);
                              #'RF'(Random Forest);'GB'(Gradient Boosting)
        
    eval_on=0                 # 0 for no; 1 for yes (display confusion matrix/classification report)
    
    output_file = "proba_" + model_type + ".csv"  # Filename for probability output 
    
    
    ## 2. Run Main Fuction
    
    main(input_train_file=input_file_train, 
         input_test_file=input_file_test,
         sample_on=sampling_on, 
         sample_type=sampling_type, 
         model_method=model_type, 
         eval_on=eval_on, 
         proba_file=output_file)

