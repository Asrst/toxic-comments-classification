## Machine Learning to Identify Toxic Comments

### Overview:

More details of the competition can be found here: 
> source: [kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

### Results:

#### 1. 01-Eda-baselines.ipynb
- [blog explaining this notebook](https://asrst.github.io/post/toxic-comments-eda-baselines/)
- Linear models are very well suited for this problem
- Logistic Regression is having better baseline score than any other model
- Logistic also has an edge with lower time/space complexity compared to ensembles and also, we can impove this further by text preprocessing & hyper parameter tuning.
- So rather than trying complex models, I will settle with linear models & will try to improve the performance using preprocessing, finetuning & feature engineering. My next target is to improve to score to more than 0.98 using only linear models.

 **Baseline-Scores**

| Model         | Public LB | Private LB | Comments |
| ------------- | --------- | ---------- | -------- |
| Naive Bayes   | 0.89144   | 0.88520    | tfidf    |
| Logistic      | 0.96877   | 0.96534    | tfidf    |
| Linear SVM    | 0.95035   | 0.94956    | tfidf    |
| Random Forest | 0.91497   | 0.93083    | tfidf    |
| XGB/LGB       | 0.91114   | 0.93331    | tfidf    |



#### 02-Improving-linear-model-baselines.ipynb

- Finally, after lots of tries, I achevied the 0.981 using the NBLogistic (Naive Bayes + Logistic) Model on one run. With K-fold cross validation that score improved a bit to 0.9816.
- In case of linear model, minimal proprocessing of text data (lowering, punctuations & stopwords removal) gave the better results. I tried few other pre-processings like lemmatization/emoji-conversion but didn't get good results.
- I experimented with xgb & lgbm if they can beat the above score, but they were not anywhere near, best being only around 0.96(xgb).

 **Improved linear models**
 
 - tfidf_word & tfidf_char features were concatenated.
 - regualarisation parameter is tuned.

| Model         | Public LB | Private LB | Comments              |
| ------------- | --------- | ---------- | ----------------------|
| Naive Bayes   | 0.90931   | 0.90696    | tfidf - (words + char)|
| Logistic      | 0.97550   | 0.97394    | tfidf - (words + char)|
| Linear SVM    | 0.96350   | 0.96956    | tfidf - (words + char)|
| NBLogistic    | 0.97819   | 0.97664    | tfidf - (words + char)|

- performed k-fold (5-folds) cross validation on NBLogisitc:

| NBLogistic    | 0.98160   | 0.98201    | tfidf - (words + char)|


#### 03-Embed + Gru+ Conv1d.ipynb

- The highest score in kaggle is 0.989...I just want know what's the effort needed to improve the score to > 0.985.Here are my observations
- Without use of pretrained embedding the scores for deep learning models are also similar to that of linear models lying around 0.98.
- In case of deep neural net models, proprocessing of text data seem to have affect especially to cross that 0.985 barrier. I took references from this [zafar's script](https://www.kaggle.com/fizzbuzz/toxic-data-preprocessing) for text pre-processing and it helped to improve my score by 0.1.
- Experimented with glove & fasttext (300-dimiensions) embedding...especially preprocessing + fasttext embeddings helped me to take the score of the 0.9845, while the 
model trained with glove scored around 0.983
- This above notebook contains the architecture on which I obtained the 0.9851 on the public leaderboard but it was down to 0.9848 on private leaderbaord.

 **deep neural nets**

| Model                                | Public LB | Private LB | Comments |
| -------------------------------------| --------- | ---------- | -------- |
| embed + gru                          | 0.98104   | 0.97952    |          |
| fasttext embed + gru                 | 0.98334   | 0.97305    |          |
| embed+gru+conv1d (minimal preprocess)| 0.98395   | 0.98353    |          |
| embed+gru+conv1d (regex preprocess)  | 0.98433   | 0.98402    |          |
| fastext embed + gru + conv1d (regex) | 0.98516   | 0.98486    |          |

#### 04-Embed + BiLstm.ipynb

- With architecture in the above notebook I crossed my target of 0.9850 on both on the public and private leaderbaords.
- blending & stacking predictions of different models whose correlation is low. And these are few references I followed to do the same
    - [mlwave](https://mlwave.com/kaggle-ensembling-guide/)
    - [kaggle blog](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)

| Model                                     | Public LB | Private LB | Comments |
| ------------------------------------------| --------- | ---------- | -------- |
| fasttext embed+Lstm                       | 0.98390   | 0.98307    |          |
| fasttext embed+2xBiLstm (minimal process) | 0.98459   | 0.98443    |          |
| fasttext embed+2xBiLstm (regex preprocess)| 0.98581   | 0.98550    |          |

- And The Best score was mean roc_auc of 0.98619 (Public LB) & 0.98602 (Private LB) from the ensemble of predictions from 7 different models & best single model is based on the Bidirectional LSTM with fasttext pretrained embeddings.

### what top scorers have done differently:

The below mentioned techniques although not very significant, known to increased scores on test-set around 0.01-0.02.(high score: mean roc-auc of 0.9889)

- Bert: This is the state of art of model for various text classification tasks. This model is mainly based on the "Transformers." & can be used for transfer learning.
- Training for more epoch with K-Fold cross validation: Although validation score is constantly increasing...training more for 50 to 100 epcohs with early stopping gave better results to some extent (around 0.01 variation in test scores).
**Although the above two steps look simple to follow, Due to hardware & kaggle kernel runtime limitaions I didn't get chance to try these.**

- Back translation: A way to augument text data. They translated the train data into different languages using translation services & then back translated them again into English. This produces slightly different texts than original.
- Test time augumentation: Same Back translation method is followed on the test data and final scores were average of predictions on different dataset variations.