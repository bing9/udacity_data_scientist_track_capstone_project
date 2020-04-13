# udacity_data_scientist_track_capstone_project
 Final Project to complete the track

Blog published: https://medium.com/@robertbingcheng/pyspark-predicting-customer-churn-e4502fd7a4ca

## Project Requirements from Udacity
##### Build Your Data Science Project
In this capstone project, you will leverage what you’ve learned throughout the program to build a data science project of your choosing. Your project deliverables are:

* A Github repository of your work.
* A blog (or other medium for a write-up) post written for a technical audience, or a deployed web application powered by data.

You'll follow the steps of the data science process that we've discussed:

1. You will first define the problem you want to solve and investigate potential solutions.
2. Next, you will analyze the problem through visualizations and data exploration to have a better understanding of what algorithms and features are appropriate for solving it.
3. You will then implement the algorithms and metrics of your choice, documenting the preprocessing, refinement, and post-processing steps along the way.
4. Afterwards, you will collect results about your findings, visualize significant quantities, validate/justify your results, and make any concluding remarks about whether your implementation adequately solves the problem.
5. Finally, you will construct a blog post (or other medium for a write-up) to document all of the steps from start to finish of your project, or deploy your results into a web application.

##### Setting Yourself Apart

An important part of landing a job or advancing your career as a data scientist is setting yourself apart through impressive data science projects. By now, you've completed several guided projects, and now's your chance to show off your skills and creativity. You'll receive a review of your project with feedback from a Udacity mentor, and they will focus on how your project demonstrates your skills as a well-rounded data scientist.

This project is designed to prepare you for delivering a polished, end-to-end solution report of a real-world problem in a field of interest. When developing new technology, or deriving adaptations of previous technology, properly documenting your process is critical for both validating and replicating your results.

Things you will learn by completing this project:

* How to research and investigate a real-world problem of interest.
* How to accurately apply specific data science algorithms and techniques.
* How to properly analyze and visualize your data and results for validity.
* How to document and write a report of your work.

## Project environment and requirements
* Release: emr-5.20.0 or later
* Applications: Spark: Spark 2.4.0 on Hadoop 2.8.5 YARN with Ganglia 3.7.2 and Zeppelin 0.8.0
* Instance type: m3.xlarge
* Number of instance: 3

## Background
For this project, we have analyzed an artificial company called Sparkify that provides streaming music services. The dataset provided contains user activity data like play next music, downgrade/upgrade service between free/paid amount. We will use it to predict whether certain is likely to leave the Sparkify platform.
## Technology
The project needs to use Spark engine. The data is hosted on Amazon S3: s3n://udacity-dsnd/sparkify/sparkify_event_data.json

* Release: emr-5.20.0 or later
* Applications: Spark: Spark 2.4.0 on Hadoop 2.8.5 YARN with Ganglia 3.7.2 and Zeppelin 0.8.0
* Instance type: m5.xlarge
* Number of instance: 3
## Problem Statement
Predicting Customer Churn using data from user activities. There are in total of 226 unique customers and the total churn rate is 23%. This is slighlty higher than the actual Spotify churn rate 19.8% in 2018. (Data from Statista.com)

Sparkify churn data
```+-----+-----+-------------+
|churn|count|pct% of total|
+-----+-----+-------------+
|    1|   52|     23%     |
|    0|  174|     77%     |
+-----+-----+-------------+
```
![Statista](/Pictures/1_XhOs8Pf0AmjWFJmBXUIo0w.png)

## Metrics
We will use the standard precision, recall, and f1 score to evaluate the model performance.
Data Exploration
For the given data on Udacity ‘mini_sparkify_event_data.json’. We have observed the below columns. Most of the columns are self-explanatory. I have attached a description of certain acronyms.

```
root
 |-- artist: string
 |-- auth: string (authorization status like logged in)
 |-- firstName: string
 |-- gender: string
 |-- itemInSession: long
 |-- lastName: string
 |-- length: double
 |-- level: string (paid or free account status)
 |-- location: string
 |-- method: string
 |-- page: string (key column to define churn)
 |-- registration: long
 |-- sessionId: long
 |-- song: string
 |-- status: long
 |-- ts: long (Timestamp)
 |-- userAgent: string (The browser used)
 |-- userId: string
 ```

## Data Visualization
We have created several visualizations, for example, the top 5 songs played by number of times played and the number of users played.

To see the distribution of all the songs played in the histogram, we created the below visualization in y-log scale: The x-axis shows the number of times each song played and the y-axis shows the accumulated unique songs: this shows it is a long tail business where 80% of songs played are only played less than 200.

![Visualization Songs Played](/Pictures/songs_played_distribution.png)

## Data Pre-processing
1. Defining churn: if the customer has the status ‘cancellation confirmation’ we will then mark this user as churn = 0 with below SQL code.
```
SELECT DISTINCT userID, 1 as churn
FROM churn_data
WHERE page = ‘Cancellation Confirmation’
```

2. Convert timestamp into a date-time format
Use Spark User Defined Functions to convert timestamp.
3. Drop Null values
Using spark function dropna()

## Implementation — Feature
What are the potential reason for churn and incorporate these features in the model to predict the customer churn:
Customer taste and availability of music

* If the customer liked/ listened to most popular music (played by a large number of people). Then this customer more likely finds songs he/she likes on the platform. This needs then below features:
* Song total played by the number of times
* Song total played by the number of people
Customer lock-in (if the customer has spent already a lot of time in Sparkify, it becomes difficult to change platform.)
* How many days have the customer been with the platform (Max timestamp — min timestamp)
* How many active days the customer with the platform (Count distinct timestamp days)
* Days active/ Days total
* Customer played songs total
Other existing features to convert to numeric
* Gender
* Free or Paid customer

## Implementation — Model

For our question, we should choose Spark ML models for classification into two classes (Churn and no Churn). Thus we have looked into the document from Spark: https://spark.apache.org/docs/latest/ml-classification-regression.html
* Logistic regression
* Decision tree classifier
* Random forest classifier
* Gradient-boosted tree classifier
* Multilayer perceptron classifier
* Linear Support Vector Machine
* One-vs-Rest classifier (a.k.a. One-vs-All)
* Naive Bayes

We thus tried to implement from the sequence introduced above from Logistic regression.

## Result and Refinement
Model 1 — Logistic regression is not providing meaningful output.
Model 2 — Decision Tree
```
pricesion is 0.61, recall is 0.85, f1 score is 0.71
```
This is still considered to be low performance comparing to our unbalanced data in a 23% churn prediction rate.
