import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt


@st.cache_resource
def load_assets():
    # Load the model
    with open('Multi_Machine_Failure_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('multi_shap_explainer.pkl', 'rb') as file:
        explainer = pickle.load(file)

    return model, explainer

@st.cache_data
def load_config():
    # Load the column order
    with open('multi_model_columns.json', 'r') as f:
        columns = json.load(f)
        
    # Load the threshold
    with open('multi_model_threshconfig.json', 'r') as f:
        threshold = json.load(f)

        
    return columns, threshold

def main():
    st.title("Multi-Label Machine Failure Prediction")

    # Load model and explainer
    model, explainer = load_assets()
    columns, threshold = load_config() 

    all_fefatures = ['Type',
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
        'TempDiff',
        'Power [W]',
        'OverstrainMetric']
    
    with st.sidebar:
        st.header("Input Features")
        input_data = {}
        
        input_data['Type'] = st.sidebar.selectbox('Product Type', ['L', 'M', 'H'])
        input_data['Air temperature [K]'] = st.sidebar.slider('Air Temperature (K)', 295.3, 304.5, 300.1, 0.1)
        input_data['Process temperature [K]'] = st.sidebar.slider('Process Temperature (K)', 305.0, 314.0, 310.2, 0.1)
        input_data['Rotational speed [rpm]'] = st.sidebar.slider('Rotational Speed (rpm)', 1160, 2900, 1500)
        input_data['Torque [Nm]'] = st.sidebar.slider('Torque (Nm)', 3.0, 80.0, 40.5, 0.1)
        input_data['Tool wear [min]'] = st.sidebar.slider('Tool wear [min]', 0, 260, 108)

    if st.button("Predict"):

        # Prepare the input data
        input_df = pd.DataFrame([input_data])

        processed_df = input_df.copy()

        st.write("Engineering new features...")
        processed_df['TempDiff'] = processed_df['Process temperature [K]'] - processed_df['Air temperature [K]']
        processed_df['Power [W]'] = processed_df['Torque [Nm]'] * (processed_df['Rotational speed [rpm]'] * 2 * np.pi / 60)
        processed_df['OverstrainMetric'] = processed_df['Tool wear [min]'] * processed_df['Torque [Nm]']

        multi_processed_df = processed_df[columns]

        st.dataframe(processed_df[['TempDiff', 'Power [W]', 'OverstrainMetric']].style.format("{:.2f}"))
            
        with st.expander("See Raw Input Data"):
            st.dataframe(input_df)

        list_of_prob_arrays = model.predict_proba(multi_processed_df )
        
        st.subheader("Failure Risk Assessment")

        all_failure = threshold.keys()
        #lets exclude rnf
        ALL_LABELS = [label for label in all_failure if label != 'RNF']
        
        for i, label_name in enumerate(ALL_LABELS):
            
            # Get the probability of "Failure" (class 1)
            prob_failure = list_of_prob_arrays[i][0, 1]
            
            # Look up the *custom* threshold for this *specific* label
            custom_threshold = threshold[label_name]
            
            # Make the decision
            is_failure = (prob_failure >= custom_threshold)
            
            # Display the result
            st.write(f"**{label_name} Risk**")
            
            if is_failure:
                st.error(f"Predicted: FAILURE (Risk: {prob_failure*100:.2f}% | Threshold: {custom_threshold*100}%)")
            else:
                st.success(f"Predicted: No Failure (Risk: {prob_failure*100:.2f}% | Threshold: {custom_threshold*100}%)")
            
            # Create a progress bar
            st.progress(prob_failure)

        st.subheader("Model Prediction Analysis (SHAP)")
        st.write("Generating 'why' plots for each failure type...")

        try:
            #Extract the pipeline's internal parts
            preprocessor = model.named_steps['preprocessor']
            classifier = model.named_steps['classifier']
            
            #Get the estimators
            estimators = classifier.estimators_
            
            # Get the fully processed data
            processed_data_transformed = preprocessor.transform(multi_processed_df)
            feature_names = preprocessor.get_feature_names_out()
            processed_df_shap = pd.DataFrame(processed_data_transformed, columns=feature_names)

            # Loop through each label and its corresponding model
            for i, label_name in enumerate(ALL_LABELS):
                
                with st.expander(f"See SHAP Waterfall for {label_name}"):
                    
                    # Get the specific simple model for this label
                    single_model = estimators[i]
                    
                    # Create the explainer for *this model*
                    individual_explainer = shap.TreeExplainer(single_model)
                    
                    # Get SHAP values
                    all_shap_values = individual_explainer.shap_values(processed_df_shap)
                    shap_values_for_one_pred = all_shap_values[0, :, 1]
                    base_value = individual_explainer.expected_value[1]
                    
                    # Create the explanation object
                    shap_explanation = shap.Explanation(
                        values=shap_values_for_one_pred, 
                        base_values=base_value, 
                        data=processed_df_shap.iloc[0], 
                        feature_names=processed_df_shap.columns.tolist()
                    )

                    # Create and display the plot
                    fig, ax = plt.subplots()
                    shap.plots.waterfall(shap_explanation, max_display=10, show=False)
                    st.pyplot(fig)
                    plt.close(fig) 

        except Exception as e:
            st.error(f"An error occurred while generating the SHAP plots: {e}")

 




        
    


if __name__ == "__main__":
    main()