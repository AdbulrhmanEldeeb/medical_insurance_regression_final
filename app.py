import streamlit as st 
import pandas as pd 
from joblib import load 
from sklearn.ensemble import RandomForestRegressor 

st.set_page_config(page_title='Insurance Prediction',page_icon='image.jpeg')
forrest=load('forrest.p')
y_sc=load('y_sc.joblib')
sc=load('sc.joblib')
st.title('prediction of insurance cost')
st.markdown(r'based on Random forrest regressor with 84% accuracy')

def main () : 
        
    with st.form('main')  :
        st.markdown('## Enter your data')
        col1,col2=st.columns(2,gap='medium')
        gender=col2.radio('Gender',options=('male','female'))
        bmi=col1.number_input(label='Body mass index (bmi)',min_value=5 ,max_value=60,value=20)
        st.divider()
        age=st.slider(label='Age',min_value=5,max_value=110,value=25)
        chlidren=col1.number_input(label='Number of Children',min_value=0,max_value=7)
        smoker=col2.radio('Are you a Smoker',options=('yes','no'),index=1)
        region=st.selectbox('Region',options=('northwest','northeast','southeast','southwest'))





        predict=st.form_submit_button('predict')
        df={'gender':gender,'bmi':bmi,'age':age,'children':chlidren,'somker':smoker,'region':region}
        if predict : 
            return df  

df=main()
# print(df)

age_1=df['age']
 
if df['gender']=='male': 
    gender_1=1 
else :
    gender_1=0    
if df['somker'] == 'yes': 
    smoker_1=1 
else : 
    smoker_1=0 

if df['region']=='southwest' :  
    result=forrest.predict(sc.transform([[0.0, 0.0, 0.0, 1.0, age_1, gender_1, int(df['bmi']), int(df['children']), smoker_1]]))

elif df['region']=='southeast' :
    result=forrest.predict(sc.transform([[0.0, 0.0, 1.0, 0.0, age_1, gender_1, int(df['bmi']), int(df['children']), smoker_1]]))

elif df['region']=='northwest' :  
    result=forrest.predict(sc.transform([[0.0, 1.0, 0.0, 0.0, age_1, gender_1, int(df['bmi']), int(df['children']), smoker_1]]))

elif df['region']=='northeast'  : 
    result=forrest.predict(sc.transform([[1.0, 0.0, 0.0, 0.0, age_1, gender_1, int(df['bmi']), int(df['children']), smoker_1]]))
            
prediction=y_sc.inverse_transform(result.reshape(-1,1))[0][0]
st.markdown('### the prediction')
st.markdown(f"your insurance cost will be   **{round(prediction,0) }$**")


