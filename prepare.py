from imports import *

df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

def prepare_HR():
    df['Attrition'] = df['Attrition'].apply(lambda x:1 if x == 'Yes' else 0)
    df['OverTime'] = df['OverTime'].apply(lambda x:1 if x == 'Yes' else 0)
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

def encode_HR_data(df):
    #capture all the categorical features 
    x_cat = df[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]
    onehotencoder = OneHotEncoder()
    x_cat = onehotencoder.fit_transform(x_cat).toarray()
    x_cat = pd.DataFrame(x_cat)
    x_numerical = df[['Age', 'Attrition', 'DailyRate',
       'DistanceFromHome', 'Education',
       'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
       'JobLevel', 'JobSatisfaction',
       'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager']]
    x_all = pd.concat([x_cat, x_numerical], axis = 1)
    return x_all