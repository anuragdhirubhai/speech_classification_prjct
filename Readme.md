**This file explains all the codes used and concepts behind the project**
**Step 1**
The project aims to classify audio files of cats and dogs using extracted MFCC features.
Audio files are loaded, MFCC features are extracted using librosa library, and stored in
a pandas dataframe. The data is split into training and testing sets, and feature scaling
is applied. A logistic regression model is then trained on the scaled training set and
evaluated on the scaled testing set using accuracy, precision, recall, and F1-score
metrics to classify the audio files.

**Step 2**
We will be using audio files of dog and cats sounds in '.wav' format
stored in folder 'cat_dog' the said file is obtained from 'Kaggle'

**Step 3**
In the project, we imports several libraries which are required for various tasks
such as data preprocessing, feature extraction, machine learning model training and evaluation,
and file handling. Here's a brief explanation of each library used in the project:
1)os: This library provides a way of using operating system-dependent functionality like
reading and writing files, and navigating directories.
2)pandas: This library is used for data manipulation and analysis. In the project,
it is used to store and manipulate the extracted features and labels.
3)numpy: This library is used for working with arrays and matrices. In the project,
it is used to store the MFCC features extracted from audio files.
4)librosa: This library is used for audio processing and feature extraction.
In the project, it is used to load audio files, and extract MFCC features from the audio files.
5)sklearn: This library is used for machine learning and data preprocessing tasks.
In the project, it is used for data splitting, feature scaling, and training and
evaluation of a logistic regression model. Specifically, LogisticRegression is
used as a classification algorithm, while train_test_split is used for splitting
the dataset, and StandardScaler is used for feature scaling. accuracy_score,
precision_score, recall_score, and f1_score are used as evaluation metrics to
assess the performance of the model.

**Explanation of key points and key words applied in project**
1.'numpy' is a library for numerical computing, providing efficient data structures
and functions for working with arrays and matrices.
2.'librosa' is a library for working with audio data, providing functions for loading,
analyzing, and manipulating audio signals.
3.'os' is a module for working with files and directories on the operating system.
4.'StandardScaler' is a class for scaling the features to have zero mean and unit variance.
5.'LogisticRegression' is a class for implementing logistic regression models for binary classification.
6.'train_test_split' is a function for splitting the data into training and testing sets.
7.'accuracy_score', 'precision_score', 'recall_score', and 'f1_score' are functions
used for evaluating the performance of the classification model.

**Step 4**
The root directory is specified as the path "audio_class_CnD/cat_dog".
The os.chdir() method is then used to change the current working directory
to the root directory, so that subsequent code can easily access the
audio files in the directory.

**Step 5**
Now By creating a list of file names, we can then loop over each file and
extract the MFCC features from them.
The code uses the os.listdir() function to get a list of file names in
the current working directory, which was set in the previous line of
code (os.chdir(root_dir)). The list of file names is stored in the file_names variable.

**Step 6**
We require an empty dataframe to store MFCC feature which we will extract from audio files,
an empty Pandas dataframe called 'final_dataset' will be used for this purpose.

**Step 7**
Now we apply a 'for' loop which iterates over each file in the list of file names
obtained using os.listdir() function. For each file name, it uses the librosa
library to load the audio file and returns two values - the audio data and the
sampling rate. The audio data is the raw audio signal stored as a one-dimensional
numpy array, and the sampling rate is the number of samples per second used to
encode the audio signal.

**Step 7**
Now we need a code that extracts the MFCC (Mel Frequency Cepstral Coefficients)
features from the audio signal using the librosa library. The mfcc function
from the librosa library is used for this purpose. y parameter in the function
represents the audio signal, sr parameter represents the sampling rate of the
audio signal, and n_mfcc represents the number of MFCC coefficients to extract.

**Step 8**
After extracting the MFCC features for each audio file, the code calculates
the mean and standard deviation of each MFCC coefficient. The mean and
standard deviation values are calculated across all the frames in
each audio file.

**Step 9**
We require to concatenate the mean and standard deviation values of
each MFCC coefficient to form a feature vector. The np.concatenate()
function is used to concatenate these arrays.

**Step 10**
label for each audio file is extracted from its file name, and it is
appended to the feature vector using the np.append() function. Then,
the feature vector is added to the final dataset using the
final_dataset.append() method. This process is repeated for all audio
files in the directory, resulting in a dataset where each row represents
an audio file and its corresponding MFCC feature vector and label.

**Step 11**
Now we split data in test and train sets.

**Step 12**
We normalize these data set

**Step 13**
We choose our model to train i.e logistic regression.

**Step 14**
We evaluate the performance of the model on test set.

**Step 15**
We print the output of the performance matrix.
**Final Output**
MFCC features extracted for 40 audio files
Accuracy: 0.8333333333333334
Precision: 0.8571428571428571
Recall: 0.8571428571428571
F1 score: 0.8571428571428571
