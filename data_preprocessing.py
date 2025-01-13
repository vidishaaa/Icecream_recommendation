# -*- coding: utf-8 -*-
""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""- Reading the Excel File
- Data is collected by us by circulating a Google Form to the students of our Institute
"""

df = pd.read_excel("/content/final eda.xlsx")

"""Displaying the first 5 rows of our dataframe"""

df.head(5)

"""Our Dataframe contains 313 rows and 10 columns"""

df.shape

"""Description of our Dataframe"""

df.describe()

"""Checking total NULL values in each column"""

df.isnull().sum()

"""Applying forward fill"""

cols = ['Age','Favourite ice cream flavour?','Gender', 'How often do you eat ice cream in a typical month?', 'Which ice cream topping do you enjoy the most?', 'Whats ur zodiac sign?']
df.loc[:,cols] = df.loc[:,cols].ffill()
df.head()

df.isnull().sum()

df.head()

"""Renaming the columns"""

df.rename(columns={'Do you prefer classic flavours (vanilla, chocolate, strawberry) or more adventurous flavours (pistachio, bungee jumping, mint)?': 'Flavor Preference'}, inplace=True)
df.rename(columns={'How often do you eat ice cream in a typical month?': 'occurence'}, inplace=True)
df.head()

"""##Finding all the unique values to perform mapping

Favourite Ice Cream Flavour
"""

df['Favourite ice cream flavour?'].nunique()

df['Favourite ice cream flavour?'].unique()

"""Do you prefer classic flavours or more adventurous flavours?"""

df['occurence'].nunique()

df['occurence'].unique()

df['Which ice cream topping do you enjoy the most?'].nunique()

df['Which ice cream topping do you enjoy the most?'].unique()

df['Whats ur zodiac sign?'].nunique()

df['Whats ur zodiac sign?'].unique()

df['Age'].nunique()

df['Age'].unique()

"""Mapping the object data type to numeric to be able to implement ML algorithms"""

#mapping the values to integer
replace_dict = {"Vanilla": 0,"Chocolate": 1,"Lemon": 2,"Strawberry": 3
               ,"Butterscotch":4,"Blackcurrent":5,"Mango":6,"American Nuts":7,"Lemon":8,"Mint":9,
               "Sprinkles":11,"Chocolate syrup":22,"Nuts (lol)":33,"Fruits":44,"Others":55,
               "i prefer classic":1,"i like to live my life on the edge":0,
               "Rarely or never":0,"1-2 times":2,"3-4 times":3,"5-6 times":5,"7 or more times (u go girl)":7,
               "Yes":1,"No":0,"Maybe":2,
               "Aries":1,"Taurus":2,"Gemini":3,"Cancer":4,"Leo":5,"Virgo":6,"Libra":7,"Scorpio":8,"Sagittarius":9,"Capricorn":10,"Aquarius":11,"Pisces":12,"idk my zodiac sign bro":13
}
df1 = df.replace(replace_dict)

gender_mapping={'Male':0,'Female':1,'Other':2}
df1['Gender']=df1['Gender'].map(gender_mapping)
age_mapping={'15-20':2,'20-25':3,'25-30':4,'35+':5,'0-10':0,'10-15':1}
df1['Age']=df1['Age'].map(age_mapping)

df1.head()

"""##Finding the relation between Age and Ice Cream Flavour"""

age_group = '15-20'
ice_cream_flavor = 'Vanilla'
# Calculating the probability that someone of the specified age will like the specified ice cream flavor
probability = len(df[(df['Age'] == age_group) & (df['Favourite ice cream flavour?'] == ice_cream_flavor)]) / len(df[df['Age'] == age_group])
print(f"The probability that someone aged {age_group} will like {ice_cream_flavor} is: {probability:.2f}")

age_group = '20-25'
ice_cream_flavor='Chocolate'
# Calculating the probability that someone of the specified age will like the specified ice cream flavor
probability=len(df[(df['Age'] == age_group) & (df['Favourite ice cream flavour?'] == ice_cream_flavor)]) / len(df[df['Age'] == age_group])
print(f"The probability that someone aged {age_group} will like {ice_cream_flavor} is: {probability:.2f}")

"""Installing Faker to generate fake names"""

!pip install faker

from faker import Faker

faker=Faker('en_IN')
df=pd.DataFrame(df)
#function to generate fake names
def generate_fake_name():
    return faker.name()
df1['Name']=df1['Name'].apply(lambda x: generate_fake_name() if pd.isna(x) else x)
df1

df1.tail(20)

df1.head()

"""The following bar graph shows the ice cream preferences split by gender."""

grouped = df.groupby(['Gender', 'Favourite ice cream flavour?']).size().unstack(fill_value=0)
#define color palette for the flavors
colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99", "#c2c2f0", "#ffb3e6", "#c2f0c2", "#ffb366"]
ax = grouped.plot(kind='bar',stacked=True,figsize=(10, 6),color=colors,width=0.8)
plt.title("Ice Cream Flavor Preference by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Ice Cream Flavor",loc='upper right')
#adding labels for each flavor count within the bar
# for i, flavor in enumerate(grouped.columns):
#     for j, count in enumerate(grouped[flavor]):
#         ax.annotate(count, (i, grouped.iloc[:j, i].sum() + count / 2), ha='center',color='black',fontsize=10)
# plt.grid(axis='y', linestyle='--',alpha=0.6)
# plt.xticks(range(len(grouped.index)), grouped.index,rotation=0)
plt.tight_layout()

plt.show()

"""The following code shows different ice cream flavours split on the preference of various zodiac groups."""

grouped = df.groupby(['Whats ur zodiac sign?', 'Favourite ice cream flavour?']).size().unstack(fill_value=0)
#define color palette for the flavors
colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99", "#c2c2f0", "#ffb3e6", "#c2f0c2", "#ffb366"]
ax = grouped.plot(kind='bar',stacked=True,figsize=(14,10),color=colors,width=0.8)
plt.title("Ice Cream Flavor Preference by Zodiac")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Ice Cream Flavor",loc='upper right')
#create a list to collect the plt.patches data
totals=[]
#find the values and append to list
for i in ax.patches:
  totals.append(i.get_height())

#set individual bar labels using above list
total=sum(totals)

#set individual bar labesl using above list
for i in ax.patches:
  #get _x pulls left or right ;get _height pushes up or down
  ax.text(i.get_x()+0.04,i.get_height()-8,\
          str(round((i.get_height()/total)*100,2))+'%',fontsize=22,color='white')
ax.text(i.get_x()+0.04,i.get_height()-8,\
          str(round((i.get_height()/total)*100,2))+'%',fontsize=22,color='white')
plt.xticks(range(len(grouped.index)), grouped.index,rotation=0)
plt.tight_layout()

plt.show()

"""CODE MODIFICATION

"""

# # Group by Zodiac Sign and Flavor
# grouped = df.groupby(['Whats ur zodiac sign?', 'Favourite ice cream flavour?']).size().unstack(fill_value=0)

# # Define color palette for the flavors
# colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99", "#c2c2f0", "#ffb3e6", "#c2f0c2", "#ffb366"]

# # Create a grouped bar plot
# plt.figure(figsize=(14, 10))
# ax = grouped.plot(kind='bar', stacked=True, color=colors, width=0.8)

# plt.title("Ice Cream Flavor Preference by Zodiac")
# plt.xlabel("Zodiac Sign")
# plt.ylabel("Count")

# # Annotate each bar with the count
# for i in range(len(grouped.index)):
#     for j, count in enumerate(grouped.iloc[i]):
#         if count > 0:
#             ax.annotate(f'{count} ({round((count / grouped.iloc[i].sum()) * 100, 2)}%)',
#                         (i, grouped.iloc[0:j, i].sum() + count / 2),
#                         ha='center', va='center', fontsize=10, rotation=30)

# # Move the legend inside the graph and reduce the font size
# ax.legend(title="Ice Cream Flavor", loc='upper left', fontsize=8)

# plt.xticks(range(len(grouped.index)), grouped.index, rotation=45)
# plt.tight_layout()
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99","#c2c2f0", "#ffb3e6", "#c2f0c2", "#ffb366",
    "#8b4513", "#aqua", "#yellowgreen", "#lightsalmon"
]
plt.figure(figsize=(10, 6))
sns.set()
ax = sns.countplot(df,x='Age',hue='Favourite ice cream flavour?', palette=colors)
plt.title("Ice Cream Flavor Preference by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
legend_labels = df['Favourite ice cream flavour?'].unique()
legend_handles = [Line2D([0], [0], color=color, label=label) for color, label in zip(colors, legend_labels)]
#LINE 2D IS USED TO CREATE CUSTOM LEGEND
ax.legend(handles=legend_handles, title="Ice Cream Flavor", loc='upper right')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



"""The following plot shows overall split of peoples favourite ice cream flavours"""

ice_cream_counts = df['Favourite ice cream flavour?'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(ice_cream_counts, labels=ice_cream_counts.index,autopct='%1.1f%%',startangle=140)
plt.title("Favorite Ice Cream Flavors")

plt.show()

filtered_df = df[(df['Gender'] == 'Male') & (df['Age'] == '20-25')]

# Count the occurrences of each ice cream flavor in the filtered DataFrame
flavor_counts = filtered_df['Favourite ice cream flavour?'].value_counts()
plt.figure(figsize=(10, 6))
ax=flavor_counts.plot(kind='bar', color='chocolate')

plt.title("Ice Cream Flavor Preference of Males Age 21")
plt.xlabel("Ice Cream Flavor")
plt.ylabel("Count")
for i, count in enumerate(flavor_counts):
    ax.text(i, count, str(count), ha='center', va='bottom', fontsize=12)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""The following portrays peoples opinions on whether they feel their zodiac is related to their favourite ice cream flavour and vice versa"""

# Define color palette
colors = ["#FF9999", "#66B2FF", "#99FF99"]
# Count the values
opinion_counts = df["Finally, do you feel that there's a connection between your zodiac sign and your ice cream flavour preferences?"].value_counts()
plt.figure(figsize=(10, 6))
opinion_counts.plot(kind='bar', color=colors)
plt.xlabel('Opinion')
plt.ylabel('Number of People')
plt.title('Do you think your favorite ice cream flavor is related to your zodiac?')
plt.xticks(rotation=45)
for i, count in enumerate(opinion_counts):
    plt.text(i, count, str(count), ha='center', va='bottom')
plt.tight_layout()
plt.show()

"""Saving the numeric dataframe as a new csv"""

df1.to_csv('example.csv',index=False)

"""Model Deployment using Random Forest Classifier

AFTER MODIFYING THE MAPPED VALUES OUR ACCURACY WAS REDUCED
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv('/content/example.csv')
X=data[['Age','Gender','Which ice cream topping do you enjoy the most?','Whats ur zodiac sign?']]
y=data['Favourite ice cream flavour?']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
clf=RandomForestClassifier(n_estimators=300,random_state=1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
# Evaluate the model
accuracy=accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
import joblib
joblib.dump(clf,'recommendation_model.pkl')
