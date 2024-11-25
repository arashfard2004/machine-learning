import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class spam_detection :
    row_mail_data= pd.read_csv('C:/Users/arash1/Desktop/data sets/mail_data.csv')
    #replace null values with null string
    mail_data =row_mail_data.where(pd.notnull(row_mail_data),' ')
    
    #lable encoding  spam =0 , ham = 1
    mail_data.loc[mail_data['Category'] == 'spam' , 'Category'] =0
    
    #I take this maildata dataframe then locate ctegory column  if value == spam  replace all value in column with 0
    mail_data.loc[mail_data['Category'] == 'ham' , 'Category'] =1
    
    #**seperating** data as text and column
    x = mail_data['Message']
    y = mail_data['Category']
    
    #spliting data to test and set data
    x_train,x_test,y_train,y_test= train_test_split(x,y , train_size=0.2, random_state=3)
    
    #feture extraction = transform the text to feuturesectors that can use as input to logestic regression
    #this method select words that iterated more
    
    feature_extraction = TfidfVectorizer(min_df=1 , stop_words='english',lowercase= True )
    # min_df = no of itrate , stopword = word that dont consider
    
 
    
    # convert y_train , y_test value as int
    y_train = y_train.astype('int')
    y_test= y_test.astype('int')
    x_train_feture= feature_extraction.fit_transform(x_train)
        
    modle = LogisticRegression()
    modle.fit(x_train_feture,y_train)
        
      
    


# creating prediction 
print("import your email text")
input_mail =[input()]
input_mail_feture = spam_detection.feature_extraction.transform(input_mail)

# make prediction
prediction =spam_detection.modle.predict(input_mail_feture)
if prediction[0] == 1:
  print('ham mail')
else :
  print('spam mail')
