# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:24:09 2023

@author: user
"""
import joblib
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# loading the three models

heart_disease_ml = joblib.load('C:/Users/user/Desktop/multiple disease prediction system/saved model/heart-disease-model.sav')

parkison_disease_ml = pickle.load(open('C:/Users/user/Desktop/multiple disease prediction system/saved model/parkison-disease-model.sav' , 'rb'))

breast_cancer_ml = pickle.load(open('C:/Users/user/Desktop/multiple disease prediction system/saved model/breast cancer-model.sav','rb'))

# creating sidebar navigation

with st.sidebar : 
    selector  = option_menu(('Multiple Disease Predictions System'),
                            
                            ['Heart Disease Prediction',
                             'Parkinson Prediction',
                             'Breast Cancer Prediction'],
                            
                            icons= ['Heart pulse','People circle','Balloon heart'],
                            
                            default_index= 0)

# heart disease prediction page
if (selector == 'Heart Disease Prediction'):
    # page title 
    st.title('Heart Disease Prediction System')
    
    col1,col2,col3 = st.columns(3)
    
    with col1 :
        age = st.number_input('age of persons',step = 1)
    with col2 :
        sex = st.number_input('sex',min_value = 0 ,max_value = 1 ,step = 1)
    with col3 :
        chest_pain = st.number_input('chest pain type',min_value = 0,max_value=4 , step = 1)
    with col1 : 
        blood_pressure = st.number_input('resting blood pressure',min_value = 0,step = 1)
    with col2 :
        serum_chol = st.number_input('serum cholesterol in mg/dl',min_value = 0,step = 1)
    with col3 :
        blood_sugar = st.number_input('fasting blood sugar > 120mg/dl',min_value = 0,max_value = 1 ,step = 1)
    with col1 :
        ele_cal = st.number_input('resting electrocardiographic results ',min_value = 0 ,max_value = 2,step = 1)
    with col2 :
        heart_rate = st.number_input('heart rate',min_value = 0,step = 1)
    with col3 :
        induced_ang = st.number_input('induced angina',min_value = 0,max_value = 1,step = 1)
    with col1 :
        oldpeak = st.number_input('old peak',step = 0.1)
    with col2 :
        slope = st.number_input('the slope of the peak',min_value = 0,step = 1)
    with col3 :
        vessel = st.number_input('number of major vessels',min_value = 0,max_value= 3,step = 1)
    with col1 :
        thal = st.number_input('thal',min_value = 0,max_value = 3,step = 1)
    
    # code for prediction
    heart_diagnosis = ''
    
    if st.button('Heart Disease Result'):
        heart_preds = heart_disease_ml.predict([[age,sex,chest_pain,blood_pressure,serum_chol,blood_sugar,ele_cal,heart_rate,induced_ang,oldpeak,slope,vessel,thal]])
        
        if (heart_preds[0] == 1) :
            heart_diagnosis = 'The patient has heart disease'
            
        else:
            'The patient does not have heart disease'
            
    st.success(heart_diagnosis)
    
    
if (selector == 'Parkinson Prediction'):
    # page title
    st.title('Parkison Disease Prediction System')
    
    col1,col2,col3,col4 = st.columns(4)
    with col1 :
        MDVP_FoHz = st.number_input('MDVP:Fo(Hz)')
    with col2 :
        MDVP_FhiHz = st.number_input('MDVP:Fhi(Hz)')
    with col3 :
        MDVP_FloHz = st.number_input('MDVP:Flo(Hz)')
    with col4 :
        MDVP_Jitter_= st.number_input('MDVP:Jitter(%)')
    with col1 :
        MDVP_JitterAbs = st.number_input('MDVP:Jitter(Abs)')
    with col2 :
        MDVP_RAP = st.number_input('MDVP(RAP)')
    with col3 :
        MDVP_PPQ = st.number_input('MDVP(PPQ)')
    with col4 :
        Jitter_DDP = st.number_input('Jitter(DDP)')
    with col1 :
        MDVP_Shimmer = st.number_input('MDVP(Shimmer)')
    with col2:
        MDVP_ShimmerdB = st.number_input('MDVP:Shimmer(dB)')
    with col3 :
        Shimmer_APQ3 = st.number_input('Shimmer(APQ3)')
    with col4 :
        Shimmer_APQ5 = st.number_input('Shimmer(APQ5)')
    with col1 :
        MDVP_APQ = st.number_input('MDVP(APQ)')
    with col2 :
        Shimmer_DDA = st.number_input('Shimmer(DDA)')
    with col3 :
        NHR = st.number_input('NHR')
    with col4 :
        HNR = st.number_input(' HNR')
    with col1 :
        RPDE = st.number_input('RPDE')
    with col2 :
        DFA = st.number_input('DFA')
    with col3 :
        spread1 = st.number_input('spread1')
    with col4 :
        spread2 = st.number_input('spread2')
    with col1 :
        D2 = st.number_input('D2')
    with col2 :
        PPE = st.number_input('PPE')
        
    park_diagnosis = ''
    
    if st.button('Parkinson disease result') :
        park_preds = parkison_disease_ml.predict([[MDVP_FoHz,MDVP_FhiHz,MDVP_FloHz,MDVP_Jitter_,MDVP_JitterAbs,MDVP_RAP,
                                                   MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_ShimmerdB,Shimmer_APQ3,Shimmer_APQ5,
                                                   MDVP_APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])
        if (park_preds[0] > 0.5) :
            park_diagnosis = 'the patient has parkinson'
        else:
            'The patient doesn\'t have parkinson'
        
    st.success(park_diagnosis)
        
        
    
if (selector == 'Breast Cancer Prediction'):
    st.title('Breast Cancer Disease Prediction System')
    
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1 :
        m_raduis = st.number_input('mean radius')
    with col2 :
        m_texture = st.number_input('mean texture')
    with col3 :
        m_para = st.number_input('mean perimeter')
    with col4 :
        m_area = st.number_input('mean area ')
    with col5 :
        m_smooth = st.number_input('mean smoothness')
    with col1 :
        m_compact = st.number_input('mean compactness')
    with col2 :
        m_conc = st.number_input('mean concavity')
    with col3 :
        m_conc_p = st.number_input('mean concave points')
    with col4 :
        m_symmetry = st.number_input('mean symmetry')
    with col5 :
        m_frac = st.number_input('mean fractal dimension')
    with col1 :
        raduis_se = st.number_input('radius SE')
    with col2 :
        text_SE = st.number_input('texture SE')
    with col3 :
        peri_SE = st.number_input('perimeter SE')
    with col4 :
        area_se = st.number_input('area SE')
    with col5 :
        smooth_se = st.number_input('smoothness SE')
    with col1: 
        comp_se = st.number_input('compactness SE')
    with col2 :
        conc_se = st.number_input('concavity SE')
    with col3 :
        conc_p_se = st.number_input('concave points SE')
    with col4 :
        symmetry_se = st.number_input('symmetry SE')
    with col5:
        frac_se = st.number_input('fractal dimension SE')
    with col1 :
        w_raduis = st.number_input('worst radius')
    with col2 :
        w_text = st.number_input('worst texture')
    with col3 :
        w_peri = st.number_input('worst perimeter')
    with col4 :
        w_area = st.number_input('worst area')
    with col5 :
        w_smooth = st.number_input('worst smoothness')
    with col1:
        w_comp = st.number_input('worst compactness')
    with col2:
        w_conc = st.number_input('worst concavity')
    with col3 :
        w_conc_p = st.number_input('worst concave points')
    with col4 :
        w_symmetry = st.number_input('worst symmetry')
    with col5:
        w_frac = st.number_input('worst fractal dimension')
    with col1 :
        tumor_size = st.number_input('Tumor size')
    with col2:
        lym_node = st.number_input('Lymph node status')
        
    breast_diag = ''
    
    if st.button('breast cancer result') :
        breast_preds = breast_cancer_ml.predict([[m_raduis,m_texture,m_para,m_area,m_smooth,m_compact, m_conc,m_conc_p,m_symmetry,m_frac
                                                  ,raduis_se,text_SE,peri_SE,area_se,smooth_se,comp_se,conc_se,conc_p_se,symmetry_se
                                                  ,frac_se,w_raduis,w_text,w_peri,w_area,w_smooth,w_comp,w_conc,w_conc_p,w_symmetry,
                                                  w_frac,tumor_size,lym_node]])
        
        if breast_preds > 0.5 :
            breast_diag = 'There might be a  recurrence before 24 months '
            
        else :
            breast_diag = 'There might not be a recurrence before 24 months'
            
    st.success(breast_diag)