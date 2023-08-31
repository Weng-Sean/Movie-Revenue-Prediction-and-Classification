#!/usr/bin/env python
# coding: utf-8

# In[335]:


# Movie folder: https://drive.google.com/drive/folders/13JVx5e_zP5ZKrpDk_yFWKa76XebMhTzG?usp=sharing
# Save the movie folder to Google Drive before run the following code

# from google.colab import drive
# drive.mount('/content/drive')


# In[336]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import colorsys
import ast
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error,f1_score, confusion_matrix, accuracy_score
from scipy.stats import percentileofscore
from sklearn.neighbors import KNeighborsClassifier

pd.options.display.max_colwidth = 100


# In[337]:


# Load CSV files
# movie_fdr_path = '/content/drive/MyDrive/movie'
# credit_df = pd.read_csv(os.path.join(movie_fdr_path, 'tmdb_5000_credits.csv'))
# movie_df = pd.read_csv(os.path.join(movie_fdr_path, 'tmdb_5000_movies.csv'))
credit_df = pd.read_csv('tmdb_5000_credits.csv')
movie_df = pd.read_csv('tmdb_5000_movies.csv')


# # Data Munging

# ## Data info

# In[338]:


credit_df.head()


# In[339]:


credit_df.info()


# In[340]:


movie_df.head()


# In[341]:


movie_df.info()


# In[342]:


credit_df.describe()


# In[343]:


movie_df.describe()


# ## Merge datasets
# 

# In[344]:


merged_df = pd.merge(movie_df, credit_df, left_on='id', right_on="movie_id")


# In[345]:


merged_df = merged_df.drop("movie_id", axis=1)


# In[346]:


merged_df.head()


# In[347]:


merged_df.columns


# ## Clean data

# In[348]:


print(merged_df.isnull().sum())


# In[349]:


# the missing data of homepage is 64%, so we can drop this column. 
# the missing data of the tagline is 17.5%, so we also want to drop this column.
# and I drop the rows that cotain NA value in runtime, release_date, and overview
merged_df = merged_df.drop(['homepage', 'tagline'], axis=1)
merged_df = merged_df.dropna(subset=['runtime', 'release_date', 'overview'])
print(merged_df.isnull().sum())


# In[350]:


#check duplication
print("movie duplicated: ",merged_df.duplicated().sum())


# In[351]:


#check outlier
plt.figure(figsize=(6,6))
sns.boxplot(y=movie_df['revenue'])
plt.title("Distribution of revenue before dropping outliers") 
plt.show()


# In[352]:


# the highest thresold
max_thresold = merged_df['revenue'].quantile(0.95)
# the lowest thresold
min_thresold = merged_df['revenue'].quantile(0.05)
# only keep the data with price between the highest thresold and lowest threso 
merged_df = merged_df[(merged_df['revenue'] < max_thresold) & (merged_df['revenue'] > min_thresold)]


# In[353]:


# figure of the distribution of price with data after dropping outliers
plt.figure(figsize=(6,6))
sns.boxplot(y=merged_df['revenue'])
plt.title("Distribution of revenue after dropping outliers") 
plt.show()


# In[354]:


#check outlier
plt.figure(figsize=(6,6))
sns.boxplot(y=movie_df['budget'])
plt.title("Distribution of budget before dropping outliers") 
plt.show()


# In[355]:


# the highest thresold
max_thresold = merged_df['budget'].quantile(0.95)
# the lowest thresold
min_thresold = merged_df['budget'].quantile(0.05)
# only keep the data with price between the highest thresold and lowest threso 
merged_df = merged_df[(merged_df['budget'] < max_thresold) & (merged_df['budget'] > min_thresold)]


# In[356]:


# figure of the distribution of price with data after dropping outliers
plt.figure(figsize=(6,6))
sns.boxplot(y=merged_df['budget'])
plt.title("Distribution of budget after dropping outliers") 
plt.show()


# In[357]:


#check outlier
plt.figure(figsize=(6,6))
sns.boxplot(y=movie_df['vote_count'])
plt.title("Distribution of vote_count before dropping outliers") 
plt.show()


# In[358]:


# the highest thresold
max_thresold = merged_df['vote_count'].quantile(0.95)
# the lowest thresold
min_thresold = merged_df['vote_count'].quantile(0.05)
# only keep the data with price between the highest thresold and lowest threso 
merged_df = merged_df[(merged_df['vote_count'] < max_thresold) & (merged_df['vote_count'] > min_thresold)]


# In[359]:


# figure of the distribution of price with data after dropping outliers
plt.figure(figsize=(6,6))
sns.boxplot(y=merged_df['vote_count'])
plt.title("Distribution of vote_count after dropping outliers") 
plt.show()


# In[360]:


merged_df


# ## Parse Json

# In[361]:


merged_df['genres'].head(1)


# In[362]:


def extract_name(data, attr = ['name']): 
    result = []
    for i in ast.literal_eval(data): 
        for a in attr:
            result.append(i[a])
    return result


# In[363]:


merged_df['genres'] = merged_df['genres'].apply(extract_name)
merged_df['keywords'] = merged_df['keywords'].apply(extract_name)
merged_df['production_companies'] = merged_df['production_companies'].apply(extract_name)
merged_df['production_countries'] = merged_df['production_countries'].apply(extract_name)
merged_df['spoken_languages'] = merged_df['spoken_languages'].apply(extract_name)
merged_df['cast'] = merged_df["cast"].apply(extract_name)


# In[364]:


merged_df[['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages', 'cast']]


# In[365]:


def get_director(data):
    for i in ast.literal_eval(data):
        if i['department'] == 'Directing':
            return i['name']

def get_producer(data):
    for i in ast.literal_eval(data):
        if i['job'] == 'Producer':
            return i['name']


# In[366]:


merged_df['Director'] = merged_df['crew'].apply(get_director)
merged_df['Producer'] = merged_df['crew'].apply(get_producer)


# In[367]:


merged_df = merged_df.drop(["crew"], axis=1)


# In[368]:


merged_df[['Director', 'Producer']]


# # Data Visualization

# ## Release Date Analysis

# In[369]:


merged_df["release_date"]


# In[370]:


merged_df["release_date_obj"] = pd.to_datetime(merged_df['release_date'], format='%Y-%m-%d')


# In[371]:


# Extract information about release year, month, and day of week
merged_df["release_month"] = merged_df['release_date_obj'].dt.month
merged_df["release_year"] = merged_df['release_date_obj'].dt.year
merged_df['release_day_of_week'] = merged_df['release_date_obj'].dt.day_name()


# In[372]:


# display the results
merged_df[["release_month", "release_year","release_day_of_week"]]


# In[373]:


month_count = []
for i in range(1,13):
    month_count.append(len(merged_df[merged_df["release_month"] == i]))


# In[374]:


# Plot the data
plt.plot(range(1,13), month_count, '-o', linewidth=2, color='#2edb82')

# Add labels and gridlines
plt.title('Monthly Movie Releases')
plt.xlabel('Month')
plt.ylabel('Number of Releases')
plt.grid(True)

# Add annotations (optional)
max_month = max(month_count)
max_index = month_count.index(max_month)
plt.annotate(f'Max: {max_month}', xy=(max_index+1, max_month), xytext=(max_index+1.5, max_month+20),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Show the plot
plt.show()


# To better understand how the data evolves over time and identify peak points, we organized the release data into lists for each month and plotted it using line charts. After analyzing the data in this way, we discovered that the month of September had the highest number of movie releases, with a total of 333 movies. This approach allowed us to easily identify the highest data points for each month and visualize the changes in the data over time.

# In[375]:


year_dic = {}
for year in merged_df["release_year"]:
    if year in year_dic:
        year_dic[year] += 1
    else:
        year_dic[year] = 1


year_dic = dict(sorted(year_dic.items()))


# In[376]:


# Create a bar chart of the number of movies released each year
plt.bar(year_dic.keys(), year_dic.values(), color="#1a9bec")

# Add axis labels and a title to the plot
plt.xlabel("Year")
plt.ylabel("Number of Movies Released")
plt.title("Number of Movies Released Each Year")

# Increase the font size of the axis labels and title
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("Number of Movies Released Each Year", fontsize=12)

# Add a grid to the plot
plt.grid(axis='y', alpha=0.5)

# Display the plot
plt.show()


# The bar chart shows that the number of movies released each year has increased steadily from the early 2000s, with a peak of 136 movies released in 2006. After this peak, there appears to be a plateau in the number of movies released, followed by a slight decrease in more recent years.
# 

# In[377]:


peak_movies = max(year_dic.values())
peak_year = [(k,v) for k,v in year_dic.items() if v == peak_movies][0][0]
print(peak_year, peak_movies)


# In[378]:


# Group the DataFrame by day of week and count the number of movies released on each day
day_of_week_counts = merged_df.groupby("release_day_of_week").count()["id"]

# Sort the days of the week in ascending order
day_of_week_counts = day_of_week_counts.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

# Create a bar chart of the number of movies released on each day of the week
day_of_week_counts.plot.bar(color = "brown")

# Add axis labels and a title to the plot
plt.xlabel("Day of the Week")
plt.ylabel("Number of Movies Released")
plt.title("Number of Movies Released by Day of the Week")

# Increase the font size of the axis labels and title
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("Number of Movies Released by Day of the Week", fontsize=12)

# Add a grid to the plot
plt.grid(axis='y', alpha=0.5)

# Display the plot
plt.show()


# The resulting plot shows that the number of movies released varies by day of the week, with a peak on Fridays and a gradual decrease in the number of releases on weekends. The lowest number of releases occurs on Mondays.
# 

# ## Movie Genre Analysis

# In[379]:


merged_df["genres"]


# In[380]:


merged_df["release_year"]


# In[381]:


# create a 2D dictionary s.t the key of outer dictionary is the year and the key of inner 
# dictionary is the genre. The inner dictionary stores the frequency of each genre occur in the dataset
genre_dic = {}
for index, row in merged_df.iterrows():
    if row['release_year'] not in genre_dic:
        genre_dic[row['release_year']] = {}
        
    for genre in row['genres']:
        if genre in genre_dic[row['release_year']]:
            genre_dic[row['release_year']][genre] += 1
        else:
            genre_dic[row['release_year']][genre] = 1


# In[382]:


all_genres = set()
for year in genre_dic:
    for genre in genre_dic[year]:
        if genre not in all_genres:
            all_genres.add(genre)

# fill zeros
for year in genre_dic:
    for genre in all_genres:
        if genre not in genre_dic[year]:
            genre_dic[year][genre] = 0

            


# In[383]:


genre_dic[2016]


# In[384]:


# # create a list of genres and years
plt.figure(figsize = (20,11), dpi=400)
# plt.figure(figsize=(10, 6), dpi=200)

data = np.zeros((len(all_genres),len(genre_dic)))
years = sorted([key for key in genre_dic])

for y_index, y in enumerate(years):
    for g_index, g in enumerate(all_genres):
        data[g_index, y_index] = genre_dic[y][g] 

# create the heatmap with aspect ratio of 2:1 and light-to-dark color scheme
plt.imshow(data, cmap='YlOrRd', aspect=2, vmin=0, vmax=data.max(), interpolation='nearest', extent=[-0.5, len(years) - 0.5, -0.5, len(all_genres) - 0.5])

# add borders to each pixel
plt.gca().set_xticks(np.arange(-0.5, len(years)), minor=True)
plt.gca().set_yticks(np.arange(-0.5, len(all_genres)), minor=True)
plt.grid(which='minor', color='#ede9df', linestyle='-', linewidth=2)

# invert the colormap
# plt.set_cmap(plt.cm.reversed('viridis'))

# add colorbar legend
plt.colorbar()

# set x-axis and y-axis labels and ticks
plt.xticks(np.arange(len(years)), years, rotation=90)
plt.yticks(np.arange(len(all_genres)), list(all_genres)[::-1])

# set title and display the plot
plt.title('Genre Popularity by Year')
plt.show()


# From the heatmap, we can see that the popularity of some genres has remained consistent over time, such as drama, comedy, and thriller. Other genres, such as fantasy and science fiction, have become more popular over the years. We can also see some fluctuations in popularity for certain genres, such as horror and romance. 

# In[385]:


# create a list of genres and years
plt.figure(figsize=(20,8))
genres = sorted(list(set([genre for year in genre_dic for genre in genre_dic[year]])))
years = sorted(list(set([year for year in genre_dic])))

# create a data matrix for the stacked area plot
data = []
for genre in genres:
    genre_data = [genre_dic[year][genre] if genre in genre_dic[year] else 0 for year in years]
    data.append(genre_data)


# define the colors for the stacked area plot
# colors = ["blue","green", "yellow", "red", "black", "lightblue", "lightred", "lightgreen", "lightyellow"]
colors = []
for i in range(1, len(genres) + 1):
    color = colorsys.hsv_to_rgb(i/len(genres), 0.8, 1)
    colors.append((color[0], color[1], color[2], 1))

# create the stacked area plot
plt.stackplot(years, data, labels=genres, colors=colors, edgecolor='black', linewidth=0.3)

# add x-axis and y-axis labels
plt.xlabel('Year')
plt.ylabel('Number of Movies')

# set title and legend
plt.title('Genre Popularity by Year')
plt.legend(loc='upper left')



# display the plot
plt.show()


# In[386]:


# Based on the heapmap, vote_average is the weakest feature correlated with revenue, whereas vote_count is the strongest feature correlated with revenue


cols = list(merged_df.columns)
cols.remove("id")
pearsoncorr = merged_df.loc[:,cols].corr(method='pearson')
sns.heatmap(pearsoncorr,
xticklabels=pearsoncorr.columns,
yticklabels=pearsoncorr.columns,
cmap='OrRd',
annot=True,
linewidth=0.5)


# #Modelling

# ## Linear Regression

# In[387]:


x = pd.DataFrame()
x["log_bud"] = merged_df["budget"].map(lambda x:np.log(x+1))
x["log_pop"] = merged_df["popularity"].map(lambda x:np.log(x+1))
x["log_vc"] = merged_df["vote_count"].map(lambda x:np.log(x+1))

y = merged_df["revenue"].map(lambda x:np.log(x+1))


# In[388]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[389]:


linear_regression_model = LinearRegression()
linear_regression_model.fit(x_train, y_train)
prediction = linear_regression_model.predict(x_test)


# In[390]:


rmse = mean_squared_error(y_test, prediction, squared=False)
print("root mean squared error: ", rmse)

# Calculate the mean squared error of the model on the testing set
mse_test = mean_squared_error(y_test, prediction)

# Calculate the residual standard error of the model on the testing set
rse_test = np.sqrt(mse_test * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1))

print("Residual Standard Error (RSE) on testing set:", rse_test)


# The result indicates that the model has relatively low errors and can be used to predict the revenue of movies in the dataset with a reasonable degree of accuracy.
# 

# In[391]:


linear_regression_model.coef_


# In[392]:


# Define data
y1 = list(y_test)
y2 = list(prediction)

# Compute line of perfect prediction
perfect_prediction = np.linspace(min(y1 + y2), max(y1 + y2), 100)

# Create plot
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(y1, y2, cmap='cool', alpha=0.8)
ax.plot(perfect_prediction, perfect_prediction, 'k--', label='Perfect Prediction')

# Add axis labels and title
ax.set_xlabel('Real values')
ax.set_ylabel('Predictions')
ax.set_title('Prediction Vs. Real values')

# Add legend
ax.legend()

# Show plot
plt.show()


# In[393]:


scores = -cross_val_score(linear_regression_model, x_train, y_train, cv=5, scoring='neg_mean_absolute_error') 
print(scores)
print(scores.mean())


# ## Logistic Regression

# In[394]:


x = pd.DataFrame()
x["log_bud"] = merged_df["budget"].map(lambda x:np.log(x+1))
x["log_pop"] = merged_df["popularity"].map(lambda x:np.log(x+1))
x["log_vc"] = merged_df["vote_count"].map(lambda x:np.log(x+1))


log_rev = merged_df["revenue"].map(lambda x:np.log(x+1))
x["target"] = [0 if z[0] < z[1] else 1 for z in zip(log_rev, x["log_bud"])]

# Balancing the class size for model performance
copied_data = x[x["target"] == 0]
x = x.append(copied_data)


y = x["target"]
x = x.drop(["target"], axis=1)



# In[395]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(x_train, y_train)

prediction = logistic_regression_model.predict(x_test)


# In[396]:


score = f1_score(y_test, prediction)
print("Fl score:", score)


# In[397]:


# Create confusion matrix
cm = confusion_matrix(y_test, prediction)

# Create heatmap with Seaborn
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[398]:


# Calculate the mean squared error of the model on the testing set
mse_test = mean_squared_error(y_test, prediction)

# Calculate the residual standard error of the model on the testing set
rse_test = np.sqrt(mse_test * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1))

print("Residual Standard Error (RSE) on testing set:", rse_test)


# In[399]:


scores = cross_val_score(logistic_regression_model, x_train, y_train, cv=5, scoring='accuracy') 
print(scores)
print(scores.mean())


# The logistic regression model achieved an F1 score of 0.778, indicating that it can accurately predict whether a movie will earn above or below its budget based on the selected features. The RSE value of 0.524 suggests that there is some level of error in the model's predictions. Overall, the model's performance is decent, but there is still room for improvement.
# 

# ## KNN Classifier
# 

# In[400]:


x = pd.DataFrame()
x["log_bud"] = merged_df["budget"].map(lambda x:np.log(x+1))
x["log_pop"] = merged_df["popularity"].map(lambda x:np.log(x+1))
x["log_vc"] = merged_df["vote_count"].map(lambda x:np.log(x+1))

y = merged_df["revenue"]

# get y label (high-revenue and low revenue)
y_label = [1 if percentileofscore(y, i) > 50 else 0 for i in y]

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y_label, test_size=0.1, random_state=42)

# Initialize empty lists for k values and accuracy
k_values = []
accuracies = []

for i in range(1, 2200, 20):
    # instantiate KNN model with k=5
    knn = KNeighborsClassifier(n_neighbors=i)

    # fit the model on training data
    knn.fit(x_train, y_train)

    # make predictions on test data
    prediction = knn.predict(x_test)

    # evaluate the model's accuracy
    accuracy = accuracy_score(y_test, prediction)
    k_values.append(i)
    accuracies.append(accuracy)
  


# In[401]:


# Plot the graph
plt.plot(k_values, accuracies)
plt.title('Model Performance for KNN Classifier')
plt.xlabel('k value')
plt.ylabel('Accuracy')

# Add annotations (optional)
max_acc = max(accuracies)
max_index = accuracies.index(max_acc)
plt.annotate(f'Max Accuracy: {max_acc:.2f}, K Value: {k_values[max_index]}', xy=(max_index+200, max_acc), xytext=(max_index+300, max_acc-0.08),
arrowprops=dict(facecolor='black', shrink=0.07, linewidth=0.01))

plt.show()


# In[402]:


knn = KNeighborsClassifier(n_neighbors=221)

# fit the model on training data
knn.fit(x_train, y_train)

# make predictions on test data
prediction = knn.predict(x_test)

# evaluate the model's accuracy
accuracy = accuracy_score(y_test, prediction)


# In[403]:


score = f1_score(y_test, prediction)
print("Fl score:", score)


# Calculate the mean squared error of the model on the testing set
mse_test = mean_squared_error(y_test, prediction)

# Calculate the residual standard error of the model on the testing set
rse_test = np.sqrt(mse_test * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1))

print("Residual Standard Error (RSE) on testing set:", rse_test)


# In[404]:


scores = cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy') 
print(scores)
print(scores.mean())


# The linear regression model has an F1 score of 0.808 and RSE of 0.437, indicating that the model has a relatively good fit to the data.

# # Conclusion

# Linear Regression Model - The model is a regression model that utilizes budget, popularity, and vote count as input features to predict movie revenue. It learns the patterns and correlations between these features and revenue from a training dataset, and can then generate revenue predictions for new movies based on these learned relationships.
# 
# Logistic Regression Model - The logistic regression model takes budget, popularity, and vote count as input features to classify movies as either profitable or not. It learns the relationship between these features and the binary outcome of profitability from a training dataset, using a logistic function to model the probability of a movie being profitable. 
# 
# KNN Model - The model takes into consideration the budget, popularity, and vote count of movies to classify them as high-revenue or low-revenue. It does this by comparing the features of a new movie to the K number of nearest neighbors in the training dataset. The label assigned to the new movie is determined by the majority class of its K nearest neighbors.
# 
# 

# In[ ]:




