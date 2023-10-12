# C-NARS
A Python tool for the classification of narratives in survey data

Background

To help researchers and policy-makers better understand how different types of self-employments contribute to older adults’ income, retirement, and quality of life, this project develops a computational method to classify self-employment narratives in survey data.
Among 17,854 job narratives in the Health and Retirement Study between 1994 and 2018, about 4,500 instances are labeled into one of three categories – Owner, Manager, and Independent - by human coders. 
A variety of machine learning algorithms are trained and tested on the labeled data in which each narrative text is pre-processed (lemmatization, stemming, etc.) and transformed into a vector of word tokens for cosine similarity calculation among narratives. 
The best-performing classification model (Gradient Boosting Trees) is applied to the entire 17,854 instances to produce probability scores of an instance being likely to belong to each of three categories. 
A total of 14,748 instances with a probability score of 0.9 or above for ‘Independent’ or with a probability score of 0.6 or above for ‘Owner’ are filtered as accurately tagged instances because they are highly likely to be assigned correct categories (97.3% for Independent and 99.0% for Owner) when evaluated on 10 random subsets (20% of 4,500 instances each) of the labeled data. 
The remaining instances are passed to manual inspection and correction before the entire data are to be used for statistical analyses. 
The classification code sets – Classification of Narratives in Survey Data (C-NARS) - are made publicly available for researchers to implement machine learning methods for classification of narratives in survey data.
