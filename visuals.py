import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import MinMaxScaler 
# Import for modeling
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
import scipy.stats as stats

df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
df['Attrition'] = df['Attrition'].apply(lambda x:1 if x == 'Yes' else 0)
df['OverTime'] = df['OverTime'].apply(lambda x:1 if x == 'Yes' else 0)
df['Over18'] = df['Over18'].apply(lambda x:1 if x == 'Yes' else 0)
df.drop(['EmployeeCount', "StandardHours", 'Over18', 'EmployeeNumber'], axis=1, inplace=True)

def split_HR_data(df):
    '''
    This function performs split on zillow data
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)
    return train, validate, test

train, validate, test = split_HR_data(df)

left_train = train[train['Attrition'] == 1]
stayed_train = train[train['Attrition'] == 0]

def univariate():
    train.hist(bins = 30, figsize = (20, 20), color= 'r')

def Environment_Satisfaction_Countplot():
    plt.figure(figsize = [8, 4])
    sns.countplot(x ='EnvironmentSatisfaction', hue ='Attrition', data = train)

def Hourly_Rate_KDE():
    plt.figure(figsize = (12, 7))

    sns.kdeplot(left_train['HourlyRate'], label = 'Employees who left', shade = True, color = 'r')
    sns.kdeplot(stayed_train['HourlyRate'], label = 'Employees who stayed', shade = True, color = 'b')

    plt.xlabel('Hourly Rate')

def Job_Level_Countplot():
    plt.figure(figsize = [8, 4])
    sns.countplot(x ='JobLevel', hue ='Attrition', data = train)

def Job_Role_Countplot():
    plt.figure(figsize = [20, 5])
    sns.countplot(x ='JobRole', hue ='Attrition', data = train)

def Stock_Option_Countplot():
    plt.figure(figsize = [20, 5])
    sns.countplot(x ='StockOptionLevel', hue ='Attrition', data = train)

def Years_At_Company_KDE():
    plt.figure(figsize = (12, 7))

    sns.kdeplot(left_train['YearsAtCompany'], label = 'Employees who left', shade = True, color = 'r')
    sns.kdeplot(stayed_train['YearsAtCompany'], label = 'Employees who stayed', shade = True, color = 'b')

    plt.xlabel('Years with Company')

def Employee_Low_Satisfaction_EnvironmentSatisfaction():
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)

    #define data
    data1 = [20, 80]
    labels1 = ['Dissatisfied', 'Okay']
    colors = sns.color_palette('coolwarm')[0:5]

    ax1.pie(data1, labels = labels1, colors = colors, autopct='%.0f%%')
    ax1.set_title('Sales Executive', fontdict = {'fontsize' : 14})

    data2 = [20, 80]
    labels2 = ['Dissatisfied', 'Okay']

    ax2.pie(data2, labels = labels2, colors = colors, autopct='%.0f%%')
    ax2.set_title("Research Scientist", fontdict = {'fontsize' : 14})

    data3 = [20, 80]
    labels3 = ['Dissatisfied', 'Okay']

    ax3.pie(data3, labels = labels3, colors = colors, autopct='%.0f%%')
    ax3.set_title("Laboratory Technician", fontdict = {'fontsize' : 14})

    data4 = [28, 72]
    labels4 = ['Dissatisfied', 'Okay']

    ax4.pie(data4, labels = labels4, colors = colors, autopct='%.0f%%')
    ax4.set_title("Research Director", fontdict = {'fontsize' : 14})


    plt.tight_layout()
    sns.set(rc = {'figure.figsize':(10,6)})

def Employee_Low_Satisfaction_EnvironmentSatisfaction2():
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)

    #define data
    data1 = [17, 83]
    labels1 = ['Dissatisfied', 'Okay']
    colors = sns.color_palette('coolwarm')[0:5]

    ax1.pie(data1, labels = labels1, colors = colors, autopct='%.0f%%')
    ax1.set_title('Healthcare Representative', fontdict = {'fontsize' : 14})

    data2 = [15, 85]
    labels2 = ['Dissatisfied', 'Okay']

    ax2.pie(data2, labels = labels2, colors = colors, autopct='%.0f%%')
    ax2.set_title("Manufacturing Director", fontdict = {'fontsize' : 14})

    data3 = [13, 87]
    labels3 = ['Dissatisfied', 'Okay']

    ax3.pie(data3, labels = labels3, colors = colors, autopct='%.0f%%')
    ax3.set_title("Sales Representative", fontdict = {'fontsize' : 14})

    data4 = [19, 81]
    labels4 = ['Dissatisfied', 'Okay']

    ax4.pie(data4, labels = labels4, colors = colors, autopct='%.0f%%')
    ax4.set_title("Human Resources", fontdict = {'fontsize' : 14})


    plt.tight_layout()
    sns.set(rc = {'figure.figsize':(10,6)})

def level_one_ResearchScientists():
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

    #define data
    data1 = [19, 81]
    labels1 = ['Quit', 'Stayed']
    colors = sns.color_palette('coolwarm')[0:5]

    ax1.pie(data1, labels = labels1, colors = colors, autopct='%.0f%%')
    ax1.set_title('Attrition Rate', fontdict = {'fontsize' : 14})

    data2 = [21, 79]
    labels2 = ['Dissatisfied', 'Okay']

    ax2.pie(data2, labels = labels2, colors = colors, autopct='%.0f%%')
    ax2.set_title("Environment Satisfaction", fontdict = {'fontsize' : 14})

    data3 = [18, 82]
    labels3 = ['Dissatisfied', 'Okay']

    ax3.pie(data3, labels = labels3, colors = colors, autopct='%.0f%%')
    ax3.set_title("Job Satisfaction", fontdict = {'fontsize' : 14})


    plt.tight_layout()
    sns.set(rc = {'figure.figsize':(10,6)})

def level_one_LaboratoryTechnicians():
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

    #define data
    data1 = [28, 72]
    labels1 = ['Quit', 'Stayed']
    colors = sns.color_palette('coolwarm')[0:5]

    ax1.pie(data1, labels = labels1, colors = colors, autopct='%.0f%%')
    ax1.set_title('Attrition Rate', fontdict = {'fontsize' : 14})

    data2 = [20, 80]
    labels2 = ['Dissatisfied', 'Okay']

    ax2.pie(data2, labels = labels2, colors = colors, autopct='%.0f%%')
    ax2.set_title("Environment Satisfaction", fontdict = {'fontsize' : 14})

    data3 = [23, 77]
    labels3 = ['Dissatisfied', 'Okay']

    ax3.pie(data3, labels = labels3, colors = colors, autopct='%.0f%%')
    ax3.set_title("Job Satisfaction", fontdict = {'fontsize' : 14})


    plt.tight_layout()
    sns.set(rc = {'figure.figsize':(10,6)})

def level_one_SalesRepresentative():
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

    #define data
    data1 = [42, 58]
    labels1 = ['Quit', 'Stayed']
    colors = sns.color_palette('coolwarm')[0:5]

    ax1.pie(data1, labels = labels1, colors = colors, autopct='%.0f%%')
    ax1.set_title('Attrition Rate', fontdict = {'fontsize' : 14})

    data2 = [14, 86]
    labels2 = ['Dissatisfied', 'Okay']

    ax2.pie(data2, labels = labels2, colors = colors, autopct='%.0f%%')
    ax2.set_title("Environment Satisfaction", fontdict = {'fontsize' : 14})

    data3 = [16, 84]
    labels3 = ['Dissatisfied', 'Okay']

    ax3.pie(data3, labels = labels3, colors = colors, autopct='%.0f%%')
    ax3.set_title("Job Satisfaction", fontdict = {'fontsize' : 14})


    plt.tight_layout()
    sns.set(rc = {'figure.figsize':(10,6)})

def level_one_HumanResources():
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

    #define data
    data1 = [30, 70]
    labels1 = ['Quit', 'Stayed']
    colors = sns.color_palette('coolwarm')[0:5]

    ax1.pie(data1, labels = labels1, colors = colors, autopct='%.0f%%')
    ax1.set_title('Attrition Rate', fontdict = {'fontsize' : 14})

    data2 = [21, 79]
    labels2 = ['Dissatisfied', 'Okay']

    ax2.pie(data2, labels = labels2, colors = colors, autopct='%.0f%%')
    ax2.set_title("Environment Satisfaction", fontdict = {'fontsize' : 14})

    data3 = [15, 85]
    labels3 = ['Dissatisfied', 'Okay']

    ax3.pie(data3, labels = labels3, colors = colors, autopct='%.0f%%')
    ax3.set_title("Job Satisfaction", fontdict = {'fontsize' : 14})


    plt.tight_layout()
    sns.set(rc = {'figure.figsize':(10,6)})

def Stock_Option_countplot():
    plt.figure(figsize = [8, 4])
    sns.countplot(x ='StockOptionLevel', hue ='Attrition', data = df)