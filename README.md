# Topic 5 -- Advanced Optimizations 

## Overview of this Repository
This repository contains all the teaching material related to **Advanced Optimizations**. The master `branch` contains the sample code for the instructor to **reference**, and the `workshop` branch contains the **empty notebooks** for the instructor and students to program in.

## For the Instructors:

This section details instructions to guide the instructor in delivering the course. The instructor should fill out the blank notebooks in the `workshop` branch according to the reference in the `master` branch (if it is possible, use two monitors so you can have the reference code opened side by side.) **There will be function calls already written in the blank notebook, please run those calls without modifications.**

Below is the curriculum of this repository, as well as the order of content to be delivered. **Be sure to familiarize yourself with the code before teaching! Feel free to explore the notebooks for course material as well as the programming exercise `lock.py`.**

This document will detail the content of material to be delivered **in order**.

### Getting set up (Jupyter Notebok)
1. Clone this repo into a working directory
2. Switch to the `workshop` branch:
   ```bash
    $ git checkout workshop
   ```
3. Create and activate a virtual environment with the command below:
    ```bash
    # MacOS/Linux
    $ python3 -m venv env 
    $ source env/bin/activate
    ```
    ```bat
    :: Windows
    \> python -m venv env 
    \> .\env\Scripts\activate
    ```
4. If you are in the virtual environment, you should see the `(env)` marker. Now, install all the dependencies:
    ```bash
    # MacOS/Linux
    $ pip install -r requirements.txt
    ```
    ```bat
    :: Windows
    \> python -m pip install -r requirements.txt
    ```
5. You are ready to go!

### Getting Started (Google Colab)
1. Clone this repo into your working directory
2. Switch to the `workshop` branch
    ```bash
    $ git checkout workshop
    ```
3. Upload whichever notebook you need to work on into Colab.
4. Drag the `colab.zip` file into Colab.
5. Unzip the file and install the dependencies using `pip` within Colab.
6. You're ready to go!


## Topic 5 -- Advanced Optimizations

- ***Please open this within Google Colab***

### Installing Dependencies
- Talk a bit about the dependencies you're using

### Big Data
- Three main points to talk about: batch gradient descent, stochastic gradient descent, and minibatch gradient descent.
- Batch Gradient Descent
  - entire training set gets fed into the network at once
  - you make one gradient step after the entire dataset
  - Explain epochs (1 epoch = 1 pass over the entire dataset)
  - Talk about the cons of batch gradient descent
    - slow training
    - requres huge amounts of memory
- Stochastic Gradient Descent
  - one gradient step per training example
  - make sure you clarify that the formal definition of SGD is different than the SGD optimizer the students are used to in torch
  - Ask students about what are some benefits and drawbacks of SGD
    - Benefits: less memory use, much faster gradient updates
    - Drawbacks: noisy gradient descent
- Mini-batch gradient descent
  - best of both worlds between batch gradient descent and SGD
  - iterate on minibatches of training examples
  - common batch sizes are 32, 64, 128, 256, 512
  - Ask students the question that is shown in the notebook
    - answer: minibatch gradient descent is smoother, thus it will be more stable around the minimum.

### Data Generator
- In this section, you will build a simple generator function that is used to split your dataset into mini-batches.
- Getting Familiar with Yield
  - paraphrase the text and write the program
  - step through the code examples slowly and explain the procedure. Comments are written to help the instructor teach.

- Data Generator
  - Build a data generator with the students, commentate on the code while programming

### Weight Initialization
- paraphrase the text 
- Why do you need to initialize the weights randomly?
  - Talk about how initializing the weights to all 0 will effectively cause all neurons on a layer to learn the same value
- Improved Weight Initialization
  - The image shows the cost of a NN over many iterations
  - The plateau at the beginning shows a slow start due to poor weight initializations
  - paraphrase through the text
  - open the link, it leads to an excellent visualization of weight initialization

### Dropout
- paraphrase the text
- talk about how Dropout is often more effective than L2 regularization
- A New Form Of Regularization
  - Explain how dropout works. 
  - Paraphrase the text
  - mention that you can still use L2 together with dropout
  - quickly commentate the image, highlighting that on every iteration, different nodes are dropped
- Implementing in PyTorch
  - be sure to let the students know that dropout goes AFTER the activation layer

### Feature Scaling II
- Paraphrase the text, as what needs to be talk about is contained in the notebook.
- Be sure to highlight the problem of internal covariate shift
- Batch Normalization
  - applying standard normalization to each layer, making training much easier and faster
  - Batch Norm has a bit of regularization effects, therefore your network requires less regularization from dropout and L2
  - Ask the students about why batch norm makes training easier
    - Answer: just like standard norm, batch norm makes the cost function much easier to traverse.

- Where to Place Batch Norm
  - This is important! this is an often debated topic in the machine learning community!
  - Batch norm goes BETWEEN linear and activation units
  - Tell the students to use the **second layout** in the image


### Optimizers
- paraphrase the text
- Talk about how when the cost function is elongated like shown in the image, gradient descent can often oscillate back and forth, slowing down training.
- Momentum
  - each new step is the average of many previous steps
  - Accelerates propagation in the desired direction, reduces oscillations.
  - Provide a short commentary on the picture
  - Explain how the $\beta_1$ parameter relates to the number of previous steps to average across.

- RMSprop
  - Similar to momentum, but this time you average across the previous gradients squared
  - dynamically scales learning rate for each parameter. In other words, each parameter has its own learning rate.
  - $\beta_2$ parameter relates to the number of previous steps to average across

- Adam
  - Paraphrase the text
  - Explain that Adam combines momentum and RMSprop
  - Commentate on the GIFs
  - Notice that Adam is not displayed in the GIFs, but state that it is faster than AdaDelta


### Hyperparameter Search


- Paraphrase the text
- General Idea
  - basically something similar to the hyperparameter evolution game in topic 3
- Grid Search
  - Searches through all combinations of Learning Rate, L2, and Dropout
  - Step through the code slowly, explaining the loop in particular
  - Not very efficient
  - Make sure students understand in every deep learning problem, each hyperparameter will effect the outcome to a certain degree. Maybe for one problem changing learning rate has a huge effect in improving the model and L2 doesn't have much effect.
  - Explain the diagram. The histograms show the relative effectiveness of each parameter. Note how gridsearch only searched through 3 different L2 and Learning rate values.
- Random Search
  - Step through the code slowly, explaining the loop in particular
  - trying many more values of each hyperparameter, thus covering a wider range of hyperparameters
  - Explain the image, note that we also used 9 combinations of learning rate and L2, but this time we are trying out 9 different learning rates and 9 L2.
  - Note that some values landed on the peaks.
  - 

## Programming Exercise -- mnist_v2.ipynb
- ***Please open this in Google Colab***
### Preprocessing Data
- The first part (loading the data) is taken from the previous topic, thus the code is already there
- Data Generator
  - Write code for this section, commentating on your code.
  - Refer to the comments to understand what each part is about
### Neural Network
- Defining Our Model
  - Same general design as before, however we are going to add batch norm and dropout
  - Write the `fit()` function as well, however note that the tensorboard portions are written already
- Test Train
  - Write the test train section just to make sure our model runs

### Hyperparameter Search
- Most of Hyperparameter Search is already written
- Change the portions within the box, as well as wherever the comments say could be changed.
- If you are testing multiple models, **make sure you delete the model, optimizer and cost function (see comments at the bottom of the cell, inside the for loop). If you don't delete it, it will accumulate and fill up the VRAM**.
- After running hyperparmeter search, you can display the `hparams` dataframe and that will list your models as well as the hyperparameters you tried, in order of validation accuracy
- You can also load up tensorboard to compare cost graphs
- Everything else is the same as topic 4.

### Optional
- save the model, then run `lock.py`. This program is identical to the one found in topic 4.
