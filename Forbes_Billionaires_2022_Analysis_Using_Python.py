# Databricks notebook source
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter
from matplotlib.pyplot import figure
import math

data = pd.read_csv("https://raw.githubusercontent.com/Krishna-97/Data_Sets/main/2022_forbes_billionaires.csv")
print(data.head())

# COMMAND ----------

print(data.isnull().sum())

# COMMAND ----------

data = data.dropna()

# COMMAND ----------

# Convert string networth to float without effecting the original column
data['newnetworth'] = pd.to_numeric(data.networth.str.replace(r"[a-zA-Z\$]",''))

# COMMAND ----------

# Top 20 Billionaires based on thier networth 
df = data.sort_values(by = ["newnetworth"],ascending=0).head(20)
plt.figure(figsize = (40,20))
sns.histplot(x="name",hue="newnetworth",data=df)
plt.show()

# COMMAND ----------

# list of billionaires whose has Jeff
def get_person(data, name = ''):
    return data[data['name'].str.contains(name, regex=False)]
get_person(data, 'Jeff')

# COMMAND ----------

# List of Billionaires whose country is india
def get_from_country(data, name = ''):
    return data[data['country'] == name]
get_from_country(data, 'India')

# COMMAND ----------

# Pie chart top 5 domians to become a billionaire
a = data["source"].value_counts().head()
index = a.index
sources = a.values
custom_colors = [ "blue","darkgreen","violet","red","darkblue"]
plt.figure(figsize=(6, 6))
plt.pie(sources, labels=index, colors=custom_colors)
central_circle = plt.Circle((0, 0), 0.6, color='white')
fig = plt.gcf()
fig.gca().add_artist(central_circle)
plt.rc('font', size=12)
plt.title("Top 5 Domains to Become a Billionaire", fontsize=20)
plt.show()

# COMMAND ----------

# Pie chart Top 5 Countries with Most Number of Billionaires
a = data["country"].value_counts().head()
index = a.index
Countries = a.values
custom_colors = ["darkgreen","grey","pink","yellow","purple"]
plt.figure(figsize=(5, 5))
plt.pie(Countries, labels=index, colors=custom_colors)
central_circle = plt.Circle((0, 0), 0.7, color='white')
fig = plt.gcf()
fig.gca().add_artist(central_circle)
plt.rc('font', size=12)
plt.title("Top 5 Countries with Most Number of Billionaires", fontsize=20)
plt.show()

# COMMAND ----------

# Country wise comparision for Billiionaire from each industries

col = "industry"
colours = ["violet", "cornflowerblue", "darkseagreen", "mediumvioletred", "blue", "mediumseagreen", "darkmagenta", "darkslateblue", "seagreen"]
Countries_list = ["United States", "India", "United Kingdom", "Japan", "France", "Canada", "Spain", "Germany", "China"]

df = data.copy()

with plt.xkcd():
    figure(num=None, figsize=(20, 12)) 
    x=1
    for Country in Countries_list:
        df["from_country"] = df['country'].fillna("").apply(lambda x : 1 if Country.lower() in x.lower() else 0)
        small = df[df["from_country"] == 1]
        genre = ", ".join(small['industry'].fillna("")).split(", ")
        # change most_common here if you want more or less industries
        tags = Counter(genre).most_common(10)
        tags = [_ for _ in tags if "" != _[0]]
        labels, values = [_[0]+"  " for _ in tags][::-1], [_[1] for _ in tags][::-1]
        if max(values)>200:
            values_int = range(0, math.ceil(max(values)), 100)
        elif max(values)>100 and max(values)<=200:
            values_int = range(0, math.ceil(max(values))+50, 50)
        else:
            values_int = range(0, math.ceil(max(values))+25, 25)
        plt.subplot(3, 3, x)
        plt.barh(labels,values, color = colours[x-1])
        plt.xticks(values_int)
        plt.title(Country)
        x+=1
    plt.suptitle('Top Industries: Number Of Billionaires In Each industry')
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# Industries Count of billionaires based on Each Industries
def count_values(data, value = ''):
    return data[value].value_counts()
industries = count_values(data, 'industry')
industries

# COMMAND ----------

column_filtering_one = ['name', 'networth', 'source', 'rank', 'industry']
column_filtering_two = ['name', 'networth', 'source', 'industry']
column_filtering_three = ['networth', 'source', 'industry']
column_filtering_country = ['name', 'networth', 'source', 'country']
def count_values(data, value = ''):
    return data[value].value_counts()

def net_worth_more_than(data, networth = 1):
    return data[data['newnetworth'] >= networth]

def get_people_multiple_sources(data):
    return data[data["source"].str.contains(",", regex=False)]

def get_from_industry(data, name = ''):
    return data[data['industry'] == name]

def get_industries_country(data, country="United States"):
    tmp = data[data['country'] == country]
    for i in tmp["industry"].unique():
        networth_sum = sum(tmp[tmp["industry"] == i]["newnetworth"])
        print(i,":", networth_sum, "billion")

def get_industry_networth(data):
    for item in data['industry'].unique():
        industry = get_from_industry(data, item)
        print(item, "networth: {:,}".format(float(sum(industry["newnetworth"]))), " billions")
        print("---")

def billionaires_in_industry(data, name = ''):
    return len(data[data['industry'] == name])

# COMMAND ----------

data

# COMMAND ----------

# All sources counted (even multiple sources)
sources = count_values(data, "source")
# Convert string networth to float without effecting the original column
data['newnetworth'] = data.newnetworth

# All people with multiple sources
people_multiple_sources = get_people_multiple_sources(data)

# All people from <country>
from_india = get_from_country(data, 'India')
from_us = get_from_country(data, 'United States')

# All people from <country> with multiple sources

from_india_multiple_sources = get_people_multiple_sources(from_india)
from_us_multiple_sources = get_people_multiple_sources(from_us)

# COMMAND ----------

#List of Billioaires whose networth more than 2 billion and has multiple sources
print(net_worth_more_than(people_multiple_sources, 2)[column_filtering_two])

# COMMAND ----------

#List of Billionaires from us whose netwoth greaterthan 20 Billion
net_worth_more_than(from_us, 20)[column_filtering_country]

# COMMAND ----------

#List of Industries wise networth of billionaires whose country is india
from_India = data[data['country'] == "India"]
get_industry_networth(from_India)

# COMMAND ----------


#List of percentage share of each industries and lowest, highest and mean values of age and rank based on each industry.

total_billions_all = np.sum(data["newnetworth"])


for item in data["industry"].unique():
    industry_bilionaires = data[data["industry"]==item][["name", "newnetworth", "country", "source", "rank", "age"]]
    
    total_billions_industry = np.sum(industry_bilionaires["newnetworth"])
    
    lowest_age_industry = np.min(industry_bilionaires["age"])
    highest_age_industry = np.max(industry_bilionaires["age"])
    mean_age_industry = round(np.mean(industry_bilionaires["age"]))
    
    lowest_rank_industry = np.max(industry_bilionaires["rank"])
    highest_rank_industry = np.min(industry_bilionaires["rank"])
    mean_rank_industry = round(np.mean(industry_bilionaires["rank"]))
    
    print(item, str(round((total_billions_industry/total_billions_all)*100, 2))+"%")
    print(item, " lowest age ", lowest_age_industry)
    print(item, " highest age ", highest_age_industry)
    print(item, " mean age ", mean_age_industry)

    print(item, " highest rank ", highest_rank_industry)
    print(item, " lowest rank ", lowest_rank_industry)
    print(item, " mean rank ", mean_rank_industry)
    
    print("")

# COMMAND ----------

print(data.groupby("industry")["newnetworth"].mean().sort_values(ascending=False))

# COMMAND ----------

pip install -U kaleido

# COMMAND ----------

df1 = data['source'].value_counts().reset_index()

fig = px.bar(df1, x = df1['index'], y = df1['source'], color =df1['source'],
             color_continuous_scale ='sunset',
             labels = {"index":"source","source":"count"})

fig.update_xaxes(tickangle=45, tickfont=dict(color='Indigo'))

fig.update_layout(title = 'Top industries of billionaires who built their own fortune',
                  title_x = 0.5,
                  title_font = dict(size = 20, family = 'Balto', color = 'Indigo'),
                  xaxis = dict(title = 'INDUSTRY'),
                  yaxis = dict(title = 'COUNT')
                 
                 )
fig.show()
fig.write_image("./fig1.png",scale = 50, width = 1500, height = 1050,engine = 'kaleido')

# COMMAND ----------

df1 = data['age'].value_counts().reset_index()

fig = px.bar(df1, x = df1['index'], y = df1['age'], color =df1['age'],
             color_continuous_scale ='rdbu',
             labels = {"index":"source","source":"count"})

fig.update_xaxes(tickangle=0, tickfont=dict(color='Indigo'))

fig.update_layout(title = 'Age of billionaires',
                  title_x = 0.5,
                  title_font = dict(size = 22, family = 'Balto', color = 'Indigo'),
                  xaxis = dict(title = 'AGE'),
                  yaxis = dict(title = 'COUNT')
                 
                 )
fig.show()
fig.write_image("./fig3.png",scale = 3, width = 800, height = 450, engine = 'kaleido')

# COMMAND ----------


