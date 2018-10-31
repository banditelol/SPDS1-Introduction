# gender_classification_challenge

## Overview

from Original Repository :
> This is the code for the gender classification challenge for 'Learn Python for Data Science #1' by @Sirajology on [YouTube](https://youtu.be/T5pRlIbr6gg). The code uses the [scikit-learn](http://scikit-learn.org/) machine learning library to train a [decision tree](https://en.wikipedia.org/wiki/Decision_tree) on a small dataset of body metrics (height, width, and shoe size) labeled male or female. Then we can predict the gender of someone given a novel set of body metrics.

First lesson from Siraj's Python from Machine Learning. All of the code here was created on Python 3.7.1 64 bit on Windows 10. If you want to run it on sandbox please visit :
> TBA : Google Collab Version | Kaggle Version

## Dependency

The code inside this repository uses only [scikit-learn](http://scikit-learn.org/stable/index.html).
But for the sake of following all the successive exercise consider setup anaconda environment.

## Usage

After making sure sci-kit learn is installed and you're using python 3.5+ just execute this command on your terminal:

```bash
python demo.py
```

## The challenge

> Find 3 more classifiers from the sci-kit learn [documentation](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) and add them to the demo.py code. 

After looping at scikit-learn cheatsheet I decided to add:

1. [Linear Support Vector classifier](http://scikit-learn.org/stable/modules/neighbors.html)
2. [Stochastic Gradient Descent ](http://scikit-learn.org/stable/modules/sgd.html#classification)
3. [K Nearest Neighbors](http://scikit-learn.org/stable/modules/neighbors.html)

to be honest at this point I don't know which is better and what each one exactly does, I'll fill the knowledge gap later.

>Train them on the same dataset and [compare their results](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html). You can determine accuracy by trying to predict testing you trained classifier on samples from the training data and see if it correctly classifies it. 

In this case I split the small dataset into even smaller dataset, I guess it's not a good practice to do it on already small dataset, but I couldn't find any big dataset on this topic so bear with it.

```python
# [height, weight, shoe_size]
# Training Dataset
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], 
[166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 
'female', 'female']

# Testing Dataset
X_test =  [[159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y_test = ['female', 'male', 'male']
```

Using those data I acquired the following output for each classifier (you can run the code to get the same result) :

```bash
Decision Tree :
Prediction : ['female' 'male' 'male']
Accuracy Score : 1.0

Linear SVC :
Prediction : ['female' 'male' 'male']
Accuracy Score : 1.0

Stochastic Gradient Descent :
Prediction : ['female' 'female' 'female']
Accuracy Score : 0.3333333333333333

K-Nearest Neighbors :
Prediction : ['female' 'male' 'male']
Accuracy Score : 1.0
```

And every time I run the code I get slightly different result especially from Linear SVC and SGD. So from this it could be said that **KNN** is better?

But the conclusion is not strong, because the evidence is not strong either, and at this point I didn't have the knowledge on how to compare the result beside from the accuracy of the prediction.

>Push your code repository to github then post it in the comments. I'll give the winner a shoutout a week from now!

Done :)