import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Create the header
st.write("""
# Would you survive on the titanic?

This app shows you your probability of survival if you were on the titanic.
""")
st.text('By Derek Lilienthal')
st.text('Link to Dataset https://www.kaggle.com/c/titanic/data')

# Header on the sidebar
st.sidebar.header('User Input')

# Function to take in all the inputs
def take_input():
    class_type = st.sidebar.select_slider('Class', options=['1','2','3'])
    sex = st.sidebar.selectbox('Sex',['Male','Female'])
    age = st.sidebar.select_slider('Age', options=range(0,100))
    sibsp = st.sidebar.select_slider('Number of Siblings/Spouses aboard the Titanic',options=range(0,10))
    parch = st.sidebar.select_slider('Number of parents/children aboard the Titanic',options=range(0,10))
    ticket = st.sidebar.radio('Ticket Type', ['Letters','Numbers'])
    fare = st.sidebar.number_input('Price paid for ticket')
    cabin = st.sidebar.selectbox('Cabin Number (U is unknown)',['A','B','C','D','E','F','G','U'])
    embarked = st.sidebar.selectbox('Port of Embarkation',['Cherbourg','Queenstown','Southampton'])
    title = st.sidebar.selectbox('Title',['Dr.','Master.','Miss.','Mr.','Mrs.','NoTitle'])

    # Convert the inputs into a dataframe
    data = {'Class':class_type,
            'Sex':sex,
            'Age':age,
            'SibSp':sibsp,
            'Parch':parch,
            'Ticket':ticket,
            'Fare':fare,
            'Cabin':cabin,
            'Embarked':embarked,
            'Title':title}

    return pd.DataFrame(data, index=[0])

# Create the dataframe from user input
user_df = take_input()

# Display the user input
st.subheader('User Input Parameters')
st.write(user_df)

### ------------------- Reading and Preprocessing the Data ------------------- ###

# Read in the data
input_file = "titanic_data.csv"
df = pd.read_csv(input_file)

# column PassengerId
# a unique numeric passenger ID; not needed
df.drop('PassengerId', axis=1, inplace=True)

# column Cabin
# use only first letter of Cabin
df['Cabin'] = df['Cabin'].str.slice(stop=1)
df['Cabin'].fillna('U', inplace=True)

# column Ticket
# use two categories: tickets containing letters and
# tickets containing only digits
df['Ticket'] = df['Ticket'].str.contains('[a-zA-Z]')

# column Embarked
# hardly any NA embarked values, so drop rows containing them
df.dropna(subset=['Embarked'], inplace=True)

# column Name
# retain only the title of the name, if present
def extract_title(s):
    titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.']
    for title in titles:
        if title in s:
            return title
    return 'NoTitle'

df['Title'] = df['Name'].apply(extract_title)
df.drop('Name', axis=1, inplace=True)

# column Age
# fill with median value
df['Age'].fillna(df['Age'].median(), inplace=True)

print(df['Embarked'].head(20))

# Convert all the non numeric values into numeric values
le_sex = LabelEncoder()
le_ticket = LabelEncoder()
le_cabin = LabelEncoder()
le_embarked = LabelEncoder()
le_title = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])
df['Ticket'] = le_ticket.fit_transform(df['Ticket'])
df['Cabin'] = le_cabin.fit_transform(df['Cabin'])
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
df['Title'] = le_title.fit_transform(df['Title'])

# Convert all data types that are not numeric (but actually are) into numeric
df['Pclass'] = df['Pclass'].astype(int)
df['Parch'] = df['Parch'].astype(int)
df['Age'] = df['Age'].astype(int)
df['Fare'] = df['Fare'].astype(float)

pd.set_option('display.max_columns', None)
print(df.head())

### ------------------- Converting user input into the right format ------------------- ###

"""
Below, i will be converting and encoding the user input into an array in order to make predictions
"""

user_result_list = []

# Add class to the list
user_result_list.append(user_df['Class'].values[0])
# Add Sex
if user_df['Sex'].values == 'Male':
    user_result_list.append(1)
else:
    user_result_list.append(0)
# Add Age
user_result_list.append(user_df['Age'].values[0])
# Add SibSp
user_result_list.append(user_df['SibSp'].values[0])
# Parch
user_result_list.append(user_df['Parch'].values[0])
# Ticket
if user_df['Ticket'].values == 'Numbers':
    user_result_list.append(0) # 0 means only numbers
else:
    user_result_list.append(1)  # 0 means only numbers
# Fare
user_result_list.append(user_df['Fare'].values[0])
# Cabin
user_result_list.append(le_cabin.transform(user_df['Cabin'])[0])
#print(user_result_list)
# Embarked
if user_df['Embarked'].values == 'S': # Southampton
    user_result_list.append(2)
elif user_df['Embarked'].values == 'Q': # Queenstown
    user_result_list.append(1)
else:
    user_result_list.append(0) # Cherbourg
# Title
user_result_list.append(le_title.transform(user_df['Title'])[0])

print(user_result_list)

### ------------------- Train the Model ------------------- ###

predictors = df.columns[1:] # Grab all the columns except for our target
target = 'Survived'
X = df[predictors].values
y = df[target].astype(int)

print(X[0])

# Create and train the model
clf = DecisionTreeClassifier(max_depth=7, max_leaf_nodes=26, min_samples_split=4, random_state=0)
clf.fit(X, y)

# Get the predictions
y_predict = clf.predict([user_result_list])
y_prob = clf.predict_proba([user_result_list])

# Display the results
if y_predict[0] == 1:
    st.subheader('You Survived!!!')
else:
    st.subheader('Did Not Survive')

# Display probabilities
st.write('Probability of Survival')
result_df = pd.DataFrame(y_prob, columns=['Did Not Survive','Survived'])
st.dataframe(result_df)

