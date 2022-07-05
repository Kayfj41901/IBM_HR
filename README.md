#### Personal Project By: Kayla Brock | Codeup | Jemison Cohort | June 30, 2022

### "_You don't build a business, you build people, and then people build the business_" -Zig Ziglar

#### A project using classification to predict attrition at a company. 

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

# I. Project Overview & EXECUTIVE SUMMARY

#### _Project Goal_ 

The goal of this report is to identify drivers of employee attrition so that supports can be put in place to reduce employee attrition. 

#### _Description_

_"According to a 2021 study by the Bureau of Labor Statistics, the average annual turnover rate is 57 percent across all industries, which accounts for both voluntary and involuntary turnover. The actual rate of turnover varies greatly by industry, however. Tech companies are at serious risk with an average turnover rate of 20.9 percent, the fourth highest overall behind retail, manufacturing and consumer goods, according to a 2019 study."_

                                                            - Kate Heinz
                                                            - https://builtin.com/recruiting/cost-of-turnover
                                                           

This project aims to uncover the main factors that lead to employee attrition. If the attrition factor is something that can be controlled for, a recommendation will be made. 

I am interested in this project because I believe employee retention is a key component of building a successful business. I feel fortunate to live in a time where this type of scientific research is even possible! In the last decade new businesses' have emerged that offer their employees excellent salaries, benefits, stock-options, and working conditions. Many of these businesses, that forewent the traditional approach of paying the CEO six figures, (and the workers 'as little as possible'), have grown into very successful companies. While this dataset is a ficticious dataset created by IBM data scientists, I believe it was carefully designed to be a good representation of actual factors of attrition(I am hoping it is loosely 'based on' their internal data). I also believe IBM is an excellent company with excellent benefits. I believe I will see, within the many features, how their benefit's package supports employee retention. Through this evidence, I hope more companies will be encouraged to invest more in their own people; knowing that what is supportive of the people is supportive of the success of the organization.

#### _Initial Thoughts & Hypotheses_

I believe employees who report a low satisfaction with their environment are more likely to quit. I believe people who are paid less are more likely to quit. I believe entry-level employees are more likely to quit. I believe certain job roles have a higher attrition rate than other job roles. I believe employees with stock options are less likely to quit. I believe employees are more likely to quit if they have only been with the company a short period of time.  

#### _Initial Questions_

- Is there a relationship between environment satisfaction and attrition?
- Is there a relationship between hourly rate and attrition? 
- Is there a relationship between Job Level and attrition? 
- Is there a relationship between Job Role and attrition? 
- Is there a relationship between stock options and attrition?
- Is there a relationship between years with company and attrition?

#### _Key Findings_

The goal of this report was to identify drivers of attrition so that supports could be put in place to lower attrition. Through data exploration and testing I have concluded that employees in the following categories are at a higher risk of leaving the company: 

- Job Level One Sales Representatives (42% of employees in this category ended up leaving the company)
- Job Level One Human Resources Employees (30% of employees in this category ended up leaving the company)
- Job Level One Laboratory Technicians (28% of employees in this category ended up leaving the company)
- It should also be noted that Environment Satisfaction was found to have a relationship with attrition and 28% of Research Directors reported a '1' (the lowest possible score) for 'Environment Satisfaction'

Three classification models were created to predict attrition: decision tree, random forest, and logistic regression. Logistic Regression, with a c-statistic of 50, was ultimately the best model. The train accuracy was 89, validate 87, and finally tested at an accuracy score of 88%. This score beat baseline by four percent. 


I recommend further investigation be done to identify the root cause of the research directors' environment dissatisfaction. (28% of Research Directors selected a 1 on a scale of 1 to 4 under the survey category 'Environment Satisfaction').I also recommend further investigation be conducted on level one sales representatives, level one human resources employees, and level one laboratory technicians. 

            

#### _Deliverables_

* README file - overview of project as well as steps to reproduce 
* Scratch Jupyter Notebook - Jupyter Notebook with ideas/exploration 
* prepare.py - contains code to prepare and split the data 
* visuals.py - contains code for visuals 
* Final Report Jupyter Notebook - contains final presentation

# II. Project Data

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

#### _Data Dictionary_

The final DataFrame used to explore the data for this project contains the following variables (columns). The variables, along with their data types, are defined below: 


```python
from collections import OrderedDict
import pandas as pd
features = OrderedDict([ ('feature', ['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
       'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
       'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
       'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']), ('datatype', ['integer', 'object', 'object', 'integer', 'object',
                                               'integer', 'integer', 'object', 'integer', 'integer', 
                                               'integer', 'object', 'integer', 'integer', 'integer', 'object',
                                               'integer', 'object', 'integer', 'integer', 'integer', 'object', 
                                               'object','integer', 'integer','integer', 'integer','integer', 
                                               'integer','integer', 'integer','integer', 'integer','integer', 'integer'])])                           

df = pd.DataFrame.from_dict(features)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>datatype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Attrition</td>
      <td>object</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BusinessTravel</td>
      <td>object</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DailyRate</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Department</td>
      <td>object</td>
    </tr>
    <tr>
      <th>5</th>
      <td>DistanceFromHome</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Education</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>7</th>
      <td>EducationField</td>
      <td>object</td>
    </tr>
    <tr>
      <th>8</th>
      <td>EmployeeCount</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>9</th>
      <td>EmployeeNumber</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>10</th>
      <td>EnvironmentSatisfaction</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Gender</td>
      <td>object</td>
    </tr>
    <tr>
      <th>12</th>
      <td>HourlyRate</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>13</th>
      <td>JobInvolvement</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>14</th>
      <td>JobLevel</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>15</th>
      <td>JobRole</td>
      <td>object</td>
    </tr>
    <tr>
      <th>16</th>
      <td>JobSatisfaction</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MaritalStatus</td>
      <td>object</td>
    </tr>
    <tr>
      <th>18</th>
      <td>MonthlyIncome</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>19</th>
      <td>MonthlyRate</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>20</th>
      <td>NumCompaniesWorked</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Over18</td>
      <td>object</td>
    </tr>
    <tr>
      <th>22</th>
      <td>OverTime</td>
      <td>object</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PercentSalaryHike</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PerformanceRating</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>25</th>
      <td>RelationshipSatisfaction</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>26</th>
      <td>StandardHours</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>27</th>
      <td>StockOptionLevel</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>28</th>
      <td>TotalWorkingYears</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>29</th>
      <td>TrainingTimesLastYear</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>30</th>
      <td>WorkLifeBalance</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>31</th>
      <td>YearsAtCompany</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>32</th>
      <td>YearsInCurrentRole</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>33</th>
      <td>YearsSinceLastPromotion</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>34</th>
      <td>YearsWithCurrManager</td>
      <td>integer</td>
    </tr>
  </tbody>
</table>
</div>



# III. Project PLAN 

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

#### The following outlines the process taken through the data science pipeline to complete this project

#### _Plan_

In the planning stage I: read project expectations, created a project outline, wrote a project goal to include how I would measure success or failure, reviewed the overview of the dataset, documented all initial thoughts, questions, and hypotheses, created a plan for completing the project, created a data dictionary to define features, created local folder and github repository.

#### _Acquire_

In the acquire stage I: created a .gitignore, obtained HR data from Kaggle and saved it in my local folder. 

#### _Prepare_

In the Prepare stage I: reviewed the dataset to see if there were any missing values(no null values), changed two categorical features(Attrition and OverTime) to  numbers('Yes' became 1, 'No' became 0), dropped columns that were not useful for data analysis (EmployeeCount, StandardHours, Over18, and EmployeeNumber), split data into train, validate, test, created prepare.py file with function to prepare data.  

#### _Explore_

In the Explore stage I: identified relationships between the target variable and features through univariate and bivariate exploration, performed six statistical tests (4 chi^2 and 2 T-test) to determine the significance of the relationship between the target variable and the feature. All tests are supported by visuals and takeaways. 

#### _Model AND Evaluate_

In the model and evaluate stage I: established baseline accuracy, trained and fit multiple models with varying algorithms and hyperparameters, compared evaluation metrics across models, evaluated best performing models using validate set, test final model on out-of-sample testing dataset, and summarized performance

#### _Deliver_

In the final stage I: prepared final notebook in Jupyter Notebook. I: wrote out my project description, introduction to include goals, created an executive summary which included all my key findings and recommendations, created headers and dividers to organize the flow of the notebook, asked and answered all questions, added summaries and supplementary markdown to guide the reader through the notebook.

# IV. Supplementary Files 

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

- prepare.py - provides code to prepare, split, and encode the data 
- visuals.py - provides code for all visuals found in the notebook 
- imports.py - provides imports 

# V. Steps to Reproduce

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

- Create a Kaggle profile, find and download 'IBM HR' dataset
- Clone this repo (including prepare.py, imports.py, and visuals.py)
- Run Final Report Jupyter notebook to view the final product
