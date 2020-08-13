# Text Classification Engine
### Classify text by Sentiment and Topic
#### "Architecting for Machine Learning with AWS", August 12-13, 2020


Take a look at the [finished app](http://polarity-reviews.s3-website-us-east-1.amazonaws.com/)

## Business Problem
Building and producing products that are actually adopted by customers and solve real problems for them is a historically challenging task. Today, imagine that you have joined the machine learning team on the Amazon e-commerce site! Your webpage is full of reviews from customers for each of your products. Your product owners want to know about a negative review *immediately*. Ideally, they'd like to know why the review was negative.

## Data Set
The dataset  comes directly from the Amazon review site. This is hosted on AWS through coursework via fast.ai https://course.fast.ai/datasets. Navigate to this page and click download for **Amazon Reviews: Polarity**. The Amazon reviews polarity dataset is constructed by taking review score 1 and 2 as negative, and 4 and 5 as positive. Samples of score 3 is ignored. In the dataset, class 1 is the negative and class 2 is the positive. Each class has 1,800,000 training samples and 200,000 testing samples.

## Step 1: Sentiment Analysis
Binary classifier using the BlazingText algorithm and 1,000,000 reviews. The labels are already provided with the dataset.
* train_accuracy: 0.9856
* validation_accuracy: 0.9151
* file: sentiment_analysis.ipynb

## Step 2: Topic Modeling
I conducted two Topic Modeling exercises, both using the Latent Dirichlet Allocation (LDA) algorithm and 50,000 reviews. I split the dataset into positive reviews and negative reviews (250K each), and then I trained the LDA model on these reviews to extract the topics and the top 10 words of each. I then saved the positive and neg datasets with the predicted topics as the new labels.
Note: I was running out of time so I took a small subset of the total dataset, in order to speed things up. It would have been better to use more data.
* 500,000 reviews (250K for each sentiment)
* Models: 2
* Topics: 20 (10 each sentiment)
* file: LDA_scikitlearn.ipynb

## Step 3: Multi-label Classification
I built two multi-label classifiers, one for the positive reviews and one of the negative reviews. I used the topics from the previous step to train my data. Again, because I was running out of time, I reduced the size of my data even further, and kept only 25,000 for each sentiment. Of these, 20K were training and 5K for validation. This small data size really killed the accuracy of my model: with more time, I'd go back and retrain with all the data.
* file: topic_classifier.ipynb

**Positive Sentiment**
* train_accuracy: 0.6062
* Number of train examples: 20000
* validation_accuracy: 0.58
* Number of validation examples: 5000

**Negative Sentiment**
* train_accuracy: 0.6071
* Number of train examples: 20000
* validation_accuracy: 0.5794
* Number of validation examples: 5000

## Step 4: Write a Lambda Function
For each of my 3 models, I deployed SageMaker endpoint. Then I wrote a Lambda function and gave it an IAM role with permission to access SageMaker. The Lambda function can accept any text, and then process it to a format that BlazingText can accept. It invokes the 3 endpoints, and then returns the predictions and probabilities.
* file: lambda_function.py

## Step 5: Create a public HTTP endpoint with API Gateway
The next step in my architecture was to build an HTTP endpoint that will allow my back-end Lambda function to talk to my front-end web app. To do this, I used another AWS service, API Gateway.

## Step 6: Use Amazon S3 to host a static web app
I wrote some simple HTML code that includes a javascript function to receive text input, and then trigger the HTTP endpoint from step 5, call the lambda function, and return the predicted values to the end user.
