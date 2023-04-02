import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle

# Data Collection and Preprocessing #
# load csv file into pandas dataframe
raw_mail_data = pd.read_csv('/home/srivaishnavi/Downloads/mail_data.csv')

# replace null/missing values with null strings
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

# label encoding: outcome of spam/not spam as 0 or 1 
mail_data.loc[mail_data['Category']=='spam', 'Category'] = 0
mail_data.loc[mail_data['Category']=='ham', 'Category'] = 1

# separate mail and labels, and divide them into features 
messages = mail_data['Message']
categories = mail_data['Category']

# split data into train and test
message_train, message_test, categories_train, categories_test = train_test_split(messages, categories, test_size=0.2, random_state=3)   # size refers to amount of data in test array (train > test data)

# Convert text data into feature vectors for logistic regression input (feature extraction)
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

message_train_features = feature_extraction.fit_transform(message_train)

pickle_extracted_features = open('feature_extraction.pkl', 'wb')
pickle.dump(feature_extraction, pickle_extracted_features)
pickle_extracted_features.close()

message_test_features = feature_extraction.transform(message_test)

categories_train = categories_train.astype('int')
categories_test = categories_test.astype('int')

# Training the model #
model = LogisticRegression()

# training the logistic regression model
model.fit(message_train_features, categories_train)

# Evaluating the model #
# predict on training data
prediction_train_data = model.predict(message_train_features)
accuracy_training = accuracy_score(categories_train, prediction_train_data)
print(accuracy_training)

#evaluate on test data
prediction_test_data = model.predict(message_test_features)
accuracy_test = accuracy_score(categories_test, prediction_test_data)
print(accuracy_test)

# test mail
input_mail = ["You have won 1000 dollars. Click to claim your reward."]
input_mail_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_mail_features)
print(prediction)

# Save model as pickle object to be used in some other code
pickle_model = open('spam_mail_detector.pkl', 'wb')   # wb = 'write binary'
pickle.dump(model, pickle_model)
pickle_model.close()
