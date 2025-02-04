# MACHINE-LEARNING-MODEL-IMPLEMENTATION


**COMPANY:** CODTECH IT SOLUTIONS

**NAME:** Nandini Patel

**INTERN ID:** CT08LEW

**DOMAIN:** Python Programmer

**BATCH DURATION:** January 10th, 2025 to february 10th, 2025

**MENTOR NAME:** NEELA SHANTOSH

This Jupyter Notebook demonstrates the process of building a machine learning model for spam email detection using a Naive Bayes classifier. The model is trained to classify text messages as either "ham" (non-spam) or "spam" based on the content of the messages.

**Key Features:**

Data Preprocessing: The dataset is cleaned by converting categorical labels (ham, spam) into numerical values (0 and 1).
TF-IDF Vectorization: Text messages are converted into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).
Model Training: The Naive Bayes algorithm is used to train the model.
Evaluation: The model’s performance is evaluated using metrics like accuracy, classification report, and confusion matrix.
Custom Message Testing: Custom test messages are provided to predict if they are spam or ham.

**Requirements:**
numpy
pandas
matplotlib
scikit-learn
seaborn

**You can install the necessary libraries by running:**

#pip install numpy pandas matplotlib scikit-learn seaborn

**Dataset:**

The dataset used for training is the famous "Spam SMS" dataset, which contains labeled messages ("ham" for non-spam and "spam" for spam messages). The dataset is loaded from a .csv file located at C:\Users\user\Downloads\spam.csv.

**Steps Involved:**

Load and Preprocess the Data: Load the dataset and map labels to numeric values.
Vectorization: Transform text data into numerical features using TF-IDF.
Model Training: Train the Naive Bayes model on the processed text data.
Evaluation: Evaluate the model using accuracy, classification report, and confusion matrix.
Custom Predictions: Test the model on custom messages to predict if they are spam or ham.

* output:
* ![Image](https://github.com/user-attachments/assets/e72db332-5c9b-4a61-a221-d8591b8b58eb)
