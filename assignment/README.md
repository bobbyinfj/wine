Clara Data Take Home
====================================================================================================
Data team take home assignment for data science roles.

<br>
<br>

Purpose
-----------------------------------------------------------------------------
Hello there! We know that the recruiting process takes time and we appreciate your completing this take home assignment. We prefer this to initial phone coding screens because it provides a lower stress environment to work on a more realistic problem at your own pace. We will use your response to this assignment both before and, should you reach the stage, during a "virtual" on-site. That in mind, this exercise should take around 1 - 3 hours. Note that there is an opportunity to discuss what you would do if you had more time.

<br>
<br>

Introduction
-----------------------------------------------------------------------------
This exercise will ask you to build a model which predicts a 1 - 10 quality score for red wine in a dataset using a series of metrics. Afterwards, you will build a very small single endpoint microservice which runs the model and returns predictions to client code. Also, we will ask a few quick questions at the end of the exercise.

**Please complete this exercise in one of the following languages:** Python, Java, Javascript, or Typescript.

<br>
<br>

Exercise
-----------------------------------------------------------------------------
This exercise comes in three parts: model, microservice, and questions.

<br>

### Model
We have generated a dataset for you in the included `takehome.avro` file ([Avro format](https://avro.apache.org/)). You should predict the `rating` attribute using the following attributes:

 - brightness
 - chlorides
 - ph
 - sugar
 - sulfates
 - acidity

You can assume that the data are given in a random order. Also assume that the `rating` comes from human raters on a Likert-like scale.

<br>

### Microservice
With a model made, create a service which responds to HTTP requests. We only need one endpoint:

`[GET] /rating/prediction.json`

Return a JSON document with the rating prediction. You can assume that the attributes are provided as URL encoded parameters like `/rating/predcition.json?ph=3&brightness=5` and you should expect all 6 parameters above to be provided. If a parameter is missing, return a 400 status code.

<br>

### Questions
Having completed the service and model, we have just a few final questions:

 - How did you choose your model?
 - How well do you expect your model to perform in practice?
 - Which of the six features above is least important?
 - What would you do if you had more time to work on this problem?

<br>
<br>

Wrapping up
-----------------------------------------------------------------------------
Thank you very much for your time. Please zip up your work along with any supporting documentation or tests and send back to us by email. Please be sure to document how you would run your solution.
