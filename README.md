# News Classification

## Abstract
News is something which we depend on to know about the outside world. As we are in the modern world, a lot of news is available in the electronic format but users are facing difficulties to access news of their interest. So this can be solved by classifying news into different categories using classifiers. We used a dataset dsjVoxArticles.tsv and extracted news from it. We classified news into five broad categories : Politics and Policy, Health Care, Business and Finance, Science and Health, Criminal Justice. We implemented six classifiers to classify this news and the classifiers are : Random Forest, Ada Boost classifier, Gradient Boost classifier, Multinomial Naïve Bayesian, Support Vector classification and Multilayered Perceptron. We used standard classification evaluation metrics such as precision, recall and f-score to evaluate. By comparing the results of all the six classifiers, we observed that Multinomial Naïve Bayes classifier worked the best.

## Introduction
Nowadays information is available to users through many sources like electronic media, digital media and many more. This data is usually available in the  unstructured form and we have a lot of ways in which this data may be converted to structured form. Text classification is actually  one of the most important task in Natural Language Processing. It is the process of classifying strings or documents into different categories, depending upon the contents of the string. Text classification methods have drawn much attention in recent years and have been widely used in many programs. Nowadays un-structured data is growing very rapidly. The un-structured information cannot be used for further processing by computers. The computers typically handle text as simple sequences of character string and are unable to provide useful information from the given text, without any process performed on it. Therefore, specific processing and preprocessing methods are required in order to extract useful patterns and information from the unstructured text. Users are facing difficulties to access news of his interest which makes it a necessity to categories news so that it could be easily accessed. Categorization means grouping that allows easier navigation among articles. Internet news needs to be divided into categories. This will help users to access the news of their interest in real-time without wasting any time. News article classification is considered as a document classification problem. Document Classification means labeling a document with predefined categories. This can be achieved as the supervised learning task of assigning documents to one or more predefined categories using machine learning. Current-day document classification for news articles poses several research challenges, due to the large number of multimodal features present in the document set and their dependencies. So, we will build a classification system which automatically assign categories to news based on manual annotations. The system is evaluated using the standard classification evaluation metrics such as precision, recall and f-score. So, using six different classifiers, we classifed news into 5 categories namely : Politics and Policy, Health Care, Business and Finance, Science and Health, Criminal Justice.

## Proposed Approach
We classify data using different classifiers. The classifiers used are : Random Forest, Ada Boost classifier, Gradient Boost classifier, Multinomial Naïve Bayesian, Support Vector classification and Multilayered Perceptron.
Random Forest Model :
Random Forest classifier grows many classification trees. Each tree is trained on a bootstrapped sample of training data and at each node the algorithm only search across a random subset of the variables to determine a split. Bootstrapping is the general technique that iteratively trains the training data set and evaluates a classifier in order to improve its performance. To classify an input vector in random forests, the vector is submitted as an input to each of the trees in the forest. Each tree gives classification, and it is said that the tree votes for that class. In the classification, the forest chooses the class having the most votes.
Ada Boost Classifier :
Ada-boost classifier combines weak classifier algorithm to form strong classifier. We combine multiple classifiers as single algorith classify the objects poorly. We combine by selecting training set at every iteration and assigning right amount of weight in final voting, we can have good accuracy score for overall classifier. It is an iterative ensemble method. It improves accuracy by combining weak learners by correcting the mistakes of the weak classifier.
Gradient Boost Classifier :
Gradient boosting is a good machine learning technique for classification problems. It produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It fits an additive model in a forward stage-wise manner. In each stage, we introduce a weak learner to compensate the shortcomings of existing weak learners. In Gradient Boosting, shortcomings are identified by gradients.
Multinomial Naïve Bayes :
The multinomial Naive Bayes classifier is a specialized version of Naive Bayes. It is suitable for classification with discrete features (e.g., word counts for text classification). It is a simple technique for constructing several classifiers such as models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. It explicitly models the word counts and adjusts the underlying calculations to deal with in.
Support Vector Classification :
Support-vector machines are supervised learning models. They are associated learning algorithms that analyze data. Support-vector machines can efficiently perform both linear classification and non-linear classification, implicitly mapping their inputs into high-dimensional feature spaces. A support-vector machine constructs a set of hyperplanes in a high-dimensional space. These can be used for classification, regression, or other tasks like outliers detection.
Multilayer Perceptron :
Multi-layer Perceptron classifier connects to a Neural Network. Unlike other classification algorithms such as Support Vectors or Naive Bayes Classifier, multi-layer perceptron classifier relies on an underlying Neural Network to perform the task of classification. It is capable to learn non-linear models. It trains using back propagation. It means, it trains using some form of gradient descent and the gradients are calculated using Backpropagation. Multi-layer perceptron classifier supports multi-class classification by applying softmax as the output function.

## Implementation & Results
Random Forest Model :
First we create a random forest and then we make a prediction from the random classifier created. First we randomly select x features from total y features. Among the x features, we calculate the node d and split the node into daughter nodes using best split. We repeat the same until we create all the trees. Then we take the test features and predict the outcome using rules of each randomly created decision tree. We calculate the votes for each predicted target and we consider the high voted predicted target as the final prediction from the random forest algorithm.
Result :
 

Ada Boost Classifier :
Initially, Adaboost selects a training subset randomly. It iteratively trains the Adaboost machine learning model. It selects the training set based on the accurate prediction of the last training. It assigns higher weight to wrong classified observations. Also, it assigns the weight to the trained classifier in each iteration according to the accuracy of the classifier. The more accurate classifier will get high weight. This process iterate until the complete training data fits without any error or until reached to the specified maximum number of estimators. To classify, perform voting. In this we use max_depth=1 to tell our model that we would like our forest to be composed of trees with a single decision node and two leaves. n_estimators refers to the total number of trees in the forest.
Result :
 
Gradient Boost Classifier :
First we optimize the loss function, then we make predictions using a weak learner and then add weak learners using additive model to minimize the loss function. In this, max_depth refers to the number of leaves of each tree  whereas n_estimators refers to the total number of trees in the ensemble. The learning_rate scales the contribution of each tree. If you set it to a low value, you will need more trees in the ensemble to fit the training set, but the overall variance will be lower.
Result :
 

Multinomial Naïve Bayes :
First, we calculate the probability of data by the class they belong, which is called the base rate. We calculate the mean and standard deviation from the dataset which are required to calculate further probabilities. Then we summarize data by class. Now, we calculate the probability or likelihood which is the conditional probability of a word occuring in a document given that the document belongs to a particular category. Finally, we calculate P(Category/Document).
P(Category/Document) = P(Category) * P(Word1/Category) * P(Word2/Category) * ………
Result :
 


Support Vector Classification :
In this, we have to find a hyperplane in an N-dimensional space (N – the number of features) that distinctly classifies the data points. This is implemented in practise using a kernel. First, we import some data and we only take first two features. Hyperplane’s dimension depends upon the number of features. If we have 2 input features, then the hyperplane is just a line and if we have 3 input features, then the hyperplane becomes a two-dimensional plane. We create an instance of support vector machine and fit out data.  As we want to plot the support vectors, we do not scale our data. Then we create a mesh to plot in. 
Result :
 


Multilayer Perceptron :
In the multilayer perceptron, there are combinations of layers which are combinations of neurons. The first layer will be the input layer, the last layer will be the output layer and all the middle layers will be the hidden layers. First, we prepare data to train on a neural network. Then the network processes the input upward activating neurons as it goes to finally produce an output value. We compare the output of the network with the expected output and we calculate the error. This error is then propagated back through the network and the weights are updated according to the amount that they contributed to the error. We use our neural network to make predictions as it is trained.
Result :
 


By comparing our results, we can say that the Multinomial Naïve Bayes Classifier works the best and especially it has good and greater f1-score. 
So, on running Multinomial Naïve Bayes on our test data, we got the following result :
 
## Conclusion and Limitations
We classified news using six classifiers : Random Forest, AdaBoost, Gradient Boost, Multinomial Naive Bayes, Support Vector, Multilayer Perceptron. By observing the results, we can conclude that Multinomial Naive Bayes classifier works the best. It is even logical for multinomial naive bayes to work the best as even we humans classify based on keywords. For example, we are likely to pedict health care if we see the keywords like hospital, doctor, fever etc. Naive Bayes scans whole dataset. It finds the probabilities of each word in headline being associated with a class and then find the probability for whole headline. So, multinomial naive Bayes is the best.
We classified news to five categories. But news can be divided into many other categories too. In our dataset, all the news which do not fall into these five categories are excluded. We used only six classifiers and out of these six, multinomial naive bayes classifier works the best. But there are many other classifiers and they might work better than multinomial naive bayes classifier. 

## References
1.	Ensemble of keyword extraction methods and classifiers in text classification by Aytu g Onan a , , Serdar Koruko glu b , Hasan Bulut b a Celal Bayar University

2.	Aung, W. T., Hla, K. H. M. S.: Random forest classifier for multicategory classification of web pages. In Services Computing Conference, 2009. APSCC 2009. IEEE AsiaPacific, pp. 372-376, IEEE (2009)

3.	A Gentle Introduction to Gradient Boosting,College of Computer and Information Science,Northeastern University

4.	https://medium.com/@Synced/how-random-forest-algorithm-works-in-machine-learning-3c0fe15b6674

5.	https://medium.com/@AI_with_Kain/understanding-of-multilayer-perceptron-mlp-8f179c4a135f

6.	https://machinelearningmastery.com/support-vector-machines-for-machine-learning/

7.	https://data.world/elenadata/vox-articles

8.	Toutanova, K.: Competitive generative models with structure learning for NLP classification tasks. In Proceedings of the 2006 Conference on Empirical Methods in Natural Language Processing, (pp. 576-584) (2006)

9.	Schmidhuber, J. Deep learning in neural networks: an overview. Neural Netw. 61, 85–117 (2015).

10.	https://machinelearningmastery.com/neural-networks-crash-course/

11.	Caropreso, M. F., Matwin, S., Sebastiani, F.: Statistical phrases in automated text categorization. Centre National de la Recherche Scientifique, Paris, France (2000)


