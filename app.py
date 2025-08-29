import streamlit as st
import pandas as pd
import pickle


try:
    with open('credit_train.rol', 'rb') as file:
        model = pickle.load(file)
    
    with open('credit_train.rol', 'rb') as columns:
        model_columns = pickle.load(columns)
except FileNotFoundError:
    st.error("Model or column files not found. Please run `train_and_save_model.py` first.")
    st.stop()



st.set_page_config(
    page_title="Loan Application Assistant",
    page_icon="💰",
    layout="wide"
)


st.title("Loan Application Data Entry Portal 🏦")
st.markdown("Enter the applicant's details below. This information will be used to assess the loan application.")
st.divider()


with st.form("loan_application_form"):
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Applicant Personal Information")
        annual_income = st.number_input("Annual Income ($)", min_value=0, value=80000, step=1000,
                                        help="Applicant's total annual income.")
        home_ownership = st.selectbox("Home Ownership", 
                                      options=['Rent', 'Home Mortgage', 'Own Home', 'HaveMortgage'],
                                      help="The applicant's home ownership status.")
        years_in_job = st.selectbox("Years in Current Job", 
                                    options=['< 1 year', '1 year', '2 years', '3 years', '4 years', 
                                             '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'],
                                    help="How long the applicant has been at their current job.")

    with col2:
        st.subheader("Loan Details")
        loan_amount = st.number_input("Current Loan Amount ($)", min_value=0, value=250000, step=1000,
                                      help="The total amount of the loan requested.")
        term = st.selectbox("Loan Term", options=['Short Term', 'Long Term'],
                            help="The duration of the loan.")
        purpose = st.selectbox("Purpose of Loan", 
                               options=['Debt Consolidation', 'Home Improvements', 'Buy a Car', 
                                        'Business Loan', 'Buy House', 'Other'],
                               help="The primary reason for the loan.")

    with col3:
        st.subheader("Credit & Financial Health")
        credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=720,
                                 help="Applicant's credit score (300-850).")
        monthly_debt = st.number_input("Monthly Debt ($)", min_value=0.0, value=1500.0, step=50.0, format="%.2f",
                                       help="Applicant's total monthly debt payments.")
        years_credit_history = st.number_input("Years of Credit History", min_value=0.0, value=15.0, step=0.5, format="%.1f",
                                               help="How long the applicant has had a credit history.")

    st.divider()

    
    col4, col5, col6 = st.columns(3)

    with col4:
        st.subheader("Account Summary")
        open_accounts = st.number_input("Number of Open Accounts", min_value=0, value=8, step=1,
                                        help="Total number of open credit accounts.")
        current_credit_balance = st.number_input("Current Credit Balance ($)", min_value=0, value=25000, step=500,
                                                 help="The total balance on all credit accounts.")
        max_open_credit = st.number_input("Maximum Open Credit ($)", min_value=0, value=75000, step=1000,
                                          help="The maximum credit line available.")

    with col5:
        st.subheader("Delinquency & Problems")
        months_delinquent = st.number_input("Months Since Last Delinquent", min_value=0, value=0, step=1,
                                            help="Months passed since the last delinquency. Use 0 if none.")
        credit_problems = st.number_input("Number of Credit Problems", min_value=0, value=0, step=1,
                                          help="Number of derogatory public records or collections.")
    
    with col6:
        st.subheader("Severe Negative Indicators")
        bankruptcies = st.number_input("Bankruptcies", min_value=0, value=0, step=1,
                                       help="Number of times the applicant has declared bankruptcy.")
        tax_liens = st.number_input("Tax Liens", min_value=0, value=0, step=1,
                                    help="Number of tax liens filed against the applicant.")

    
    submitted = st.form_submit_button("Submit Application Data", type="primary")


if submitted:
    
    application_data = {
        'Current Loan Amount': loan_amount,
        'Term': term,
        'Credit Score': credit_score,
        'Annual Income': annual_income,
        'Years in current job': years_in_job,
        'Home Ownership': home_ownership,
        'Purpose': purpose,
        'Monthly Debt': monthly_debt,
        'Years of Credit History': years_credit_history,
        'Months since last delinquent': months_delinquent,
        'Number of Open Accounts': open_accounts,
        'Number of Credit Problems': credit_problems,
        'Current Credit Balance': current_credit_balance,
        'Maximum Open Credit': max_open_credit,
        'Bankruptcies': bankruptcies,
        'Tax Liens': tax_liens
    }

    
    input_df = pd.DataFrame([application_data])
    
    st.success("Application data captured successfully!")
    st.markdown("### Captured Applicant Data")
    st.dataframe(input_df)

    input_df = pd.DataFrame(input_df)

   
    input_df_encoded = pd.get_dummies(input_df)
    
   
    final_df = input_df_encoded.reindex(columns=model_columns, fill_value=0)


    approval_prob = model.predict_proba(final_df)[0][1] # Probability of class '1' (Fully Paid)

  
    st.subheader("Prediction Result")
    
    if approval_prob > 0.65:
        st.success(f"High Chance of Approval: {approval_prob:.2%}")
        st.balloons()
    elif approval_prob > 0.40:
        st.warning(f"Moderate Chance of Approval: {approval_prob:.2%}")
    else:
        st.error(f"Low Chance of Approval: {approval_prob:.2%}")
        
    st.metric(label="Approval Probability", value=f"{approval_prob:.2%}")

