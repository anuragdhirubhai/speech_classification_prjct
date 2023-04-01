from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import librosa
import numpy as np

# Set the root directory to the audio files
root_dir = "audio_class_CnD/cat_dog"
os.chdir(root_dir)

# Get a list of file names
file_names = os.listdir()

# Initialize an empty dataframe for storing the MFCC features
final_dataset = pd.DataFrame()

# Loop over each audio file and extract MFCC features
for file_name in file_names:
    # Load the audio file using librosa
    audio, sampling_rate = librosa.load(file_name)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=13)

    # Calculate the mean and standard deviation of each MFCC coefficient
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # Concatenate the mean and standard deviation values to form a feature vector
    feature_vector = np.concatenate((mfccs_mean, mfccs_std), axis=0)

    # Append the feature vector and the label to the final dataset
    label = file_name.split("_")[0]
    feature_vector = np.append(feature_vector, label)
    final_dataset = final_dataset.append(pd.Series(feature_vector), ignore_index=True)

# Rename the columns of the final dataset
columns = ["mfcc_" + str(i) for i in range(1, 27)] + ["label"]
final_dataset.columns = columns

print("MFCC features extracted for", len(file_names), "audio files")


# Split the final dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(final_dataset.drop('label', axis=1),
                                                    final_dataset['label'],
                                                    test_size=0.3,
                                                    random_state=42)

# Normalize the feature variables
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Choose a machine learning algorithm and train the model on the training set
model = LogisticRegression(random_state=42)
model.fit(X_train_norm, y_train)

# Evaluate the performance of the trained model on the testing set
y_pred = model.predict(X_test_norm)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='cat')
recall = recall_score(y_test, y_pred, pos_label='cat')
f1 = f1_score(y_test, y_pred, pos_label='cat')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
