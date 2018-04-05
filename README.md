# Battle-of-the-Sexes-ParkRun
A comparison of the data of Males and Females

## Battle of the Sexes


The aim of this post is to compare the performance and attendence of male and female athletes.
It's done with an eye toward being able to predict the gender of an athlete based on their finish time, position etc. I'm looking for variables which show separation between the two genders.


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
%matplotlib inline
from statsmodels.tsa.seasonal import seasonal_decompose
```


    


```python
path_to_file = 'C:\Users\Administrator\Documents\Python Scripts\examplepark.csv'
data = pd.read_csv(path_to_file)
data['Time'] = ((pd.to_numeric(data['Time'].str.slice(0,2)))*60)+(pd.to_numeric\
(data['Time'].str.slice(3,5)))+((pd.to_numeric(data['Time'].str.slice(6,8)))/60)
data['Date'] = pd.to_datetime(data['Date'],errors='coerce', format='%d-%m-%Y')
data['Age_Cat'] = pd.to_numeric(data['Age_Cat'].str.slice(2,4),errors='coerce', downcast='integer')
data['Age_Grade'] = pd.to_numeric(data['Age_Grade'].str.slice(0,5),errors='coerce')
```




<div>


</div>



### Adding the relative position variable.

Position isn't a reliable indicator of performance if attendence is changing. Someone could post the same time two weeks in a row but have dramatically different positons. Relative Postion is more likely to stay the same. It could change, of course, if a large group of either very fast or slow people turn up. That seems unlikely though.

Relative position is a persons position divided by the total runner count. 0 means they came in first place and 1 means last place.


```python
ser = data.groupby(['Date']).count()['Name']
Rel_Pos = []
c=0
for i in range(len(ser)):
    for j in range(ser[i]):
        rel_pos = float(data['Pos'][c])/float(ser[i])
        Rel_Pos.append(rel_pos)
        c+=1
data['Rel_Pos'] = Rel_Pos
data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Pos</th>
      <th>Name</th>
      <th>Time</th>
      <th>Age_Cat</th>
      <th>Age_Grade</th>
      <th>Gender</th>
      <th>Gen_Pos</th>
      <th>Club</th>
      <th>Note</th>
      <th>Total_Runs</th>
      <th>Run_No.</th>
      <th>Rel_Pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-11-10</td>
      <td>1</td>
      <td>Michael MCSWIGGAN</td>
      <td>18.316667</td>
      <td>35.0</td>
      <td>73.43</td>
      <td>M</td>
      <td>1.0</td>
      <td>Portmarnock Athletic Club</td>
      <td>First Timer!</td>
      <td>29.0</td>
      <td>1</td>
      <td>0.006289</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012-11-10</td>
      <td>2</td>
      <td>Alan FOLEY</td>
      <td>18.433333</td>
      <td>30.0</td>
      <td>71.16</td>
      <td>M</td>
      <td>2.0</td>
      <td>Raheny Shamrock AC</td>
      <td>First Timer!</td>
      <td>99.0</td>
      <td>1</td>
      <td>0.012579</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-11-10</td>
      <td>3</td>
      <td>Matt SHIELDS</td>
      <td>18.533333</td>
      <td>55.0</td>
      <td>85.07</td>
      <td>M</td>
      <td>3.0</td>
      <td>North Belfast Harriers</td>
      <td>First Timer!</td>
      <td>274.0</td>
      <td>1</td>
      <td>0.018868</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-11-10</td>
      <td>4</td>
      <td>David GARGAN</td>
      <td>18.650000</td>
      <td>40.0</td>
      <td>73.73</td>
      <td>M</td>
      <td>4.0</td>
      <td>Raheny Shamrock AC</td>
      <td>First Timer!</td>
      <td>107.0</td>
      <td>1</td>
      <td>0.025157</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012-11-10</td>
      <td>5</td>
      <td>Paul SINTON-HEWITT</td>
      <td>18.900000</td>
      <td>50.0</td>
      <td>79.28</td>
      <td>M</td>
      <td>5.0</td>
      <td>Ranelagh Harriers</td>
      <td>First Timer!</td>
      <td>369.0</td>
      <td>1</td>
      <td>0.031447</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = data[:1000]
df = df.drop(['Date'], axis=1)
df = df.drop(['Note'], axis=1)
df = df.drop(['Club'], axis=1)
df = df.drop(['Name'], axis=1)
df = df.drop(['Run_No.'], axis=1)
df = df.drop(['Age_Grade'], axis=1)
df = df.drop(['Age_Cat'], axis=1)
df = df.drop(['Gen_Pos'], axis=1)

df_time = df.drop(['Pos'], axis=1)
df_time = df_time.drop(['Total_Runs'], axis=1)
df_time = df_time.drop(['Rel_Pos'], axis=1)

df_rel = df.drop(['Pos'], axis=1)
df_rel = df_rel.drop(['Total_Runs'], axis=1)
df_rel = df_rel.drop(['Time'], axis=1)
```

### Using a swarmplot to check for separation


```python
df = pd.melt(df, 'Gender', var_name='measurement')
sns.swarmplot(x="measurement", y="value", hue="Gender", data=df)
```









![png](/img/output_8_1.png)


To make the plot clear I have only plotted the first 1000 records.

Each dot corresponds to a different record (race run). The colour refers to the gender.

There clearly some separation in the betwen male and female athletes in the Position category.
Let's get a closer look at Time and Relative Postion.

#### Finish time (in minutes)


```python
df_time = pd.melt(df_time, 'Gender', var_name='measurement')
sns.swarmplot(x="measurement", y="value", hue="Gender", data=df_time)
```









![png](/img/output_11_1.png)


#### Relative Postion


```python
df_rel = pd.melt(df_rel, 'Gender', var_name='measurement')
sns.swarmplot(x="measurement", y="value", hue="Gender", data=df_rel)
```









![png](/img/output_13_1.png)


In both finish times and relative position there is clear separation between the two genders. This means we should be able to use these variables to predict the gender of a runner.


```python
df1 = data.groupby(['Gender','Age_Cat']).mean()['Time']
df1.unstack(level=0)
```




    Gender  Age_Cat
    F       10.0       32.879202
            11.0       31.058809
            15.0       31.332648
            18.0       28.495502
            20.0       30.145129
    Name: Time, dtype: float64



### Finish Times versus Age Category for Men and Women.

I want see how that separation changes across age. Do men detoriate faster with age than women leading to a closing gap in times?


```python
x = [10,11,15,18,20,25,30,35,40,45,50,55,60,65,70,75,80,85]
ymax = data.groupby(['Gender','Age_Cat']).max()['Time'].unstack(level=0)
ymean = data.groupby(['Gender','Age_Cat']).mean()['Time'].unstack(level=0)
ymin = data.groupby(['Gender','Age_Cat']).min()['Time'].unstack(level=0)
ystd = data.groupby(['Gender','Age_Cat']).std()['Time'].unstack(level=0)
```


```python
plt.figure(figsize=(12,12))
plt.subplot(221)
plt.plot(x,ymax, marker = 'o')
plt.xlabel('Age Category', fontsize=18)
plt.ylabel('Max Finish Time', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.grid()

plt.subplot(222)
plt.plot(x,ymean, marker = 'o')
plt.xlabel('Age Category', fontsize=18)
plt.ylabel('Mean Finish Time', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.tight_layout()
plt.grid()

plt.subplot(223)
plt.plot(x,ymin, marker = 'o')
plt.xlabel('Age Category', fontsize=18)
plt.ylabel('Min Finish Time', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.grid()

plt.subplot(224)
plt.plot(x,ystd, marker = 'o')
plt.xlabel('Age Category', fontsize=18)
plt.ylabel('StD Finish Time', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.tight_layout()
plt.grid()
```


![png](/img/output_19_0.png)


The min times for both genders remains flat until about 50 years of age. It then rises dramatically, though at a slightly slower rate for men. 
The mean times for women rises very slowly for women up to about 65 years of age. It looks like men  see a steeper and earlier (I should probably check this rigouroussly) rise in their mean times with age.

The most interesting thing about the mean times, is the drop/increase in the female/male mean time at 18 years old. Not sure why this is and I haven't looked into it. It does coincide with a much smaller ratio of men to women than compared to other ages. But might be unrelated.

A little unexpectedly the standard devaition for men is consistently lower, showing that men have a more similar ability to one another.

Below is a boxplot which shows the (more or less) the same information as above but in a more condensed form. More info on boxplots http://www.physics.csbsju.edu/stats/box2.html


```python
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.boxplot(ax=ax, data=data, x="Age_Cat", y="Time", hue='Gender')
```




    




![png](/img/output_22_1.png)


I also wanted to see how relative position changed with age. Similar trends...


```python
fig, ax = plt.subplots(figsize=a4_dims)
sns.boxplot(ax=ax, data=data, x="Age_Cat", y="Rel_Pos", hue='Gender')
```




 




![png](/img/output_24_1.png)


### Some distributions
Next I'm going to show the distributions of Age, Finish time, Position, and Relative Position for the two genders. I'm showing variables which will help us predict the gender of am athlete.
I'm using rounded values to make the plots clearer, for calculation I'll use the raw data.


```python
df2 = data.dropna()
df2['Age_Cat'] = df2['Age_Cat'].apply(lambda x: int(x))
df2['Rounded_Time'] = df2['Time'].apply(lambda x: x//2)
df2['Rounded_Time'] = df2['Rounded_Time'].apply(lambda x: int(x*2))
df2['Rounded Pos'] = df2['Pos'].apply(lambda x: x//20)
df2['Rounded Pos'] = df2['Rounded Pos'].apply(lambda x: int(x*20))
df2['Rel_Rounded_Pos'] = df2['Rel_Pos'].apply(lambda x: int(x*25))
df2['Rel_Rounded_Pos'] = df2['Rel_Rounded_Pos'].apply(lambda x: float(x)/25)

df2.head()
```



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Pos</th>
      <th>Name</th>
      <th>Time</th>
      <th>Age_Cat</th>
      <th>Age_Grade</th>
      <th>Gender</th>
      <th>Gen_Pos</th>
      <th>Club</th>
      <th>Note</th>
      <th>Total_Runs</th>
      <th>Run_No.</th>
      <th>Rel_Pos</th>
      <th>Rounded_Time</th>
      <th>Rounded Pos</th>
      <th>Rel_Rounded_Pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-11-10</td>
      <td>1</td>
      <td>Michael MCSWIGGAN</td>
      <td>18.316667</td>
      <td>35</td>
      <td>73.43</td>
      <td>M</td>
      <td>1.0</td>
      <td>Portmarnock Athletic Club</td>
      <td>First Timer!</td>
      <td>29.0</td>
      <td>1</td>
      <td>0.006289</td>
      <td>18</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012-11-10</td>
      <td>2</td>
      <td>Alan FOLEY</td>
      <td>18.433333</td>
      <td>30</td>
      <td>71.16</td>
      <td>M</td>
      <td>2.0</td>
      <td>Raheny Shamrock AC</td>
      <td>First Timer!</td>
      <td>99.0</td>
      <td>1</td>
      <td>0.012579</td>
      <td>18</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-11-10</td>
      <td>3</td>
      <td>Matt SHIELDS</td>
      <td>18.533333</td>
      <td>55</td>
      <td>85.07</td>
      <td>M</td>
      <td>3.0</td>
      <td>North Belfast Harriers</td>
      <td>First Timer!</td>
      <td>274.0</td>
      <td>1</td>
      <td>0.018868</td>
      <td>18</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-11-10</td>
      <td>4</td>
      <td>David GARGAN</td>
      <td>18.650000</td>
      <td>40</td>
      <td>73.73</td>
      <td>M</td>
      <td>4.0</td>
      <td>Raheny Shamrock AC</td>
      <td>First Timer!</td>
      <td>107.0</td>
      <td>1</td>
      <td>0.025157</td>
      <td>18</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012-11-10</td>
      <td>5</td>
      <td>Paul SINTON-HEWITT</td>
      <td>18.900000</td>
      <td>50</td>
      <td>79.28</td>
      <td>M</td>
      <td>5.0</td>
      <td>Ranelagh Harriers</td>
      <td>First Timer!</td>
      <td>369.0</td>
      <td>1</td>
      <td>0.031447</td>
      <td>18</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Attendence by age
First here's the attendence versus age. We can see the curve is centred at about 35/40 yers of age.


```python
ax = sns.factorplot("Age_Cat", data=df2,kind='count',palette='BuPu',size=5, aspect=1.5)
```


![png](/img/output_28_0.png)


Let's break it down by gender. We can see that wider age range of men attend. While it's most popular with women between the ages of 30 and 40.


```python
ax = sns.factorplot("Age_Cat", data=df2,kind='count', col="Gender",size=5, aspect=1.4)
```


![png](/img/output_30_0.png)


#### Comparing finish times
The distribution of finish times has a lognormal shape with a peak at 25-ish minutes.


```python
ax = sns.factorplot("Rounded_Time", data=df2,kind='count',palette='GnBu_d',size=5, aspect=1.5)
```


![png](/img/output_32_0.png)


If we split this into the genders again we get the following. The average male time is roughly 25 minutes, while the female average is roughly 30 minutes. We can do a two sample t-test on this to see if are in fact distinct and this isn't a chance observation. 


```python
ax = sns.factorplot("Rounded_Time", data=df2,kind='count', row="Gender",size=5, aspect=1.5)
```


![png](/img/output_34_0.png)


#### T-test
No surprise given the amount of data we have. The means are without a doubt distinct from each other.


```python
M = data[data['Gender']=='M']
F = data[data['Gender']=='F']
print('Male mean time:') 
print(np.mean(M['Time']))
print('Female mean time:')
print(np.mean(F['Time']))
stats.ttest_ind(F['Time'], M['Time'], equal_var=False)
```

    Male mean time:
    24.7518982128
    Female mean time:
    30.3943157574
    
    Ttest_indResult(statistic=151.5950893861521, pvalue=0.0)



#### Distribution of positions
We find what resembles a pareto distribution with a large shoulder.

Splitting by gender we can see that the overall distribution is made of the male/pareto distribution and the female/normal distribution.


```python
ax = sns.factorplot("Rounded Pos", data=df2,kind='count',palette='GnBu_d', size=5, aspect=1.5)
ax.set_xticklabels(rotation=90)
```









![png](/img/output_38_1.png)



```python
ax = sns.factorplot("Rounded Pos", data=df2,kind='count', row="Gender",size=5, aspect=1.5)
ax.set_xticklabels(rotation=90)
```




    <seaborn.axisgrid.FacetGrid at 0x1875ad30>




![png](/img/output_39_1.png)


#### Distribution of Relative Position
The  relative position has been split into 25 boxes of width 0.04.

I'm not sure why distribution of the relative position has the shape it does. I would have thought it would have been flat and split evenly across each box. It might be due to the rounding process but I'm not certain. Any thoughts? Get in touch.

Splitting by gender we see two very distinct distributions.


```python
ax = sns.factorplot("Rel_Rounded_Pos", data=df2,kind='count',palette='GnBu_d',size=5, aspect=1.5)
ax.set_xticklabels(rotation=90)
```









![png](/img/output_41_1.png)



```python
ax = sns.factorplot("Rel_Rounded_Pos", data=df2,kind='count', row="Gender",size=5, aspect=1.5)
ax.set_xticklabels(rotation=90)
```









![png](/img/output_42_1.png)


### Global Variables

So far in this post we've seen that it should be possible to predict an athletes gender using the 'individual' variables; Time, Position etc.

However, we might be able to improve our predictions if we use some 'global' variables. For example the attendence that day, or the mean finish time etc. These should also help with our predictions.

The following shows the correlation of the gender ratio (males to females) with some of these global variables.


```python
dd = {'Runner Count': data.groupby('Date').size(), \
     'Min Time': data.groupby('Date').min()['Time'], \
    'Mean Time': data.groupby('Date').mean()['Time'], \
    'Max Time': data.groupby('Date').max()['Time'], \
    'STD Time': data.groupby('Date').std()['Time'], \
     'Mean_Age_Grade': data.groupby('Date').mean()['Age_Grade'], \
     'Mean_Age_Cat': data.groupby('Date').mean()['Age_Cat']}
dfdate = pd.DataFrame(data=dd)
dfdate['Ratio'] = dft['Ratio']
dfdate['Date'] = dfdate.index
dfdate.index = range(275)
dfdate.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Max Time</th>
      <th>Mean Time</th>
      <th>Mean_Age_Cat</th>
      <th>Mean_Age_Grade</th>
      <th>Min Time</th>
      <th>Runner Count</th>
      <th>STD Time</th>
      <th>Ratio</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>44.783333</td>
      <td>27.830778</td>
      <td>36.406667</td>
      <td>54.344490</td>
      <td>18.316667</td>
      <td>159</td>
      <td>5.495270</td>
      <td>0.851852</td>
      <td>2012-11-10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48.000000</td>
      <td>26.322277</td>
      <td>35.361386</td>
      <td>56.680495</td>
      <td>16.050000</td>
      <td>216</td>
      <td>5.815160</td>
      <td>1.376471</td>
      <td>2012-11-17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>51.150000</td>
      <td>26.752964</td>
      <td>35.826087</td>
      <td>55.790763</td>
      <td>16.400000</td>
      <td>268</td>
      <td>5.177254</td>
      <td>1.073770</td>
      <td>2012-11-24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44.516667</td>
      <td>25.932583</td>
      <td>35.288288</td>
      <td>57.174247</td>
      <td>16.716667</td>
      <td>236</td>
      <td>5.258924</td>
      <td>1.466667</td>
      <td>2012-12-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>52.150000</td>
      <td>25.786577</td>
      <td>35.013423</td>
      <td>57.871724</td>
      <td>17.233333</td>
      <td>162</td>
      <td>6.002006</td>
      <td>1.614035</td>
      <td>2012-12-08</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = dfdate['Ratio']
std_time = dfdate['STD Time']
mean_time = dfdate['Mean Time']
runner_count = dfdate['Runner Count']
mean_age_grade = dfdate['Mean_Age_Grade']
```


```python
plt.figure(figsize=(12,12))
plt.subplot(221)
plt.scatter(std_time,y)
plt.xlabel('STD Time', fontsize=18)
plt.ylabel('Ratio', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.legend(fontsize=16)
plt.grid()

plt.subplot(222)
plt.scatter(mean_time,y)
plt.xlabel('Mean Time', fontsize=18)
plt.ylabel('Ratio', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.legend(fontsize=16)
plt.tight_layout()
plt.grid()

plt.subplot(223)
plt.scatter(runner_count,y)
plt.xlabel('Runner Count', fontsize=18)
plt.ylabel('Ratio', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.legend(fontsize=16)
plt.grid()

plt.subplot(224)
plt.scatter(mean_age_grade,y)
plt.xlabel('Mean Age Grade', fontsize=18)
plt.ylabel('Ratio', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.legend(fontsize=16)
plt.tight_layout()
plt.grid()
```


![png](/img/output_46_0.png)


There is clear correlation between the ratio and the standard deviation and mean of finish times. As well as the attendence(runner count) and the average age of those running. Adding these 'global' variables to the individual ones should improve our prediction accuracy.

*__Note: The larger the ratio the more males. A ratio of 1 means the split was 50/50.__*

### Ratio by date
It turns out the gender ratio is also correlated to the date. This is likely because as I showed in a previous post, that attendence is at a minimum in December. So, we see a peak in the ratio in December.


```python
dft = data.groupby(['Date','Gender']).count()['Pos']
dft = dft.unstack()
dft['Ratio'] = dft['M']/dft['F']
dft['Date'] = dft.index

series = dft['Ratio']
result = seasonal_decompose(series,\
                            model='additive',freq=52)
fig = result.plot()
fig.set_size_inches(15,6)
```


![png](/img/output_49_0.png)


### Conclusion

That's all for this post. The next post will demonstrate several machine learning models to predict gender. The models will use the variables shown here to have some predictive power.

*Same Bat-time, same Bat-place!*

### Post Script

I'll leave you with a pairplot of some of the variables.


```python
df = data[:5000]
df = df.drop('Club',1)
df = df.dropna()
sns.pairplot(df, hue="Gender")
```









![png](/img/output_53_1.png)

