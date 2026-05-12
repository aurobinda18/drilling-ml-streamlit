import streamlit as st
import pandas as pd
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Drilling Quality Prediction",
    page_icon="🛢️",
    layout="wide"
)

st.title("🛢️ Drilling Quality Prediction Dashboard")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    data_path = os.path.join("data", "drilling_data.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        st.error("Data file not found!")
        return None

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "drilling_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        st.warning("Model not trained yet. Please train the model first.")
        return None

# Main app
def main():
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select Page", ["Data Overview", "Prediction", "About"])
    
    if page == "Data Overview":
        st.header("Drilling Data Overview")
        data = load_data()
        if data is not None:
            st.subheader("Dataset Preview")
            st.dataframe(data.head(10))
            
            st.subheader("Dataset Statistics")
            st.write(data.describe())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(data))
            with col2:
                st.metric("Features", len(data.columns) - 1)
    
    elif page == "Prediction":
        st.header("Predict Drilling Quality")
        model = load_model()
        
        if model is None:
            st.info("Please train the model first by running: `python train_model.py`")
        else:
            st.subheader("Input Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                depth = st.number_input("Depth (ft)", min_value=0.0, value=1000.0)
                rop = st.number_input("Rate of Penetration (ft/hr)", min_value=0.0, value=45.0)
                wob = st.number_input("Weight on Bit (klbs)", min_value=0.0, value=25.0)
                torque = st.number_input("Torque (klbs-ft)", min_value=0.0, value=12.0)
            
            with col2:
                rpm = st.number_input("RPM", min_value=0.0, value=120.0)
                flow_rate = st.number_input("Flow Rate (gpm)", min_value=0.0, value=450.0)
                spp = st.number_input("Standpipe Pressure (psi)", min_value=0.0, value=2500.0)
            
            if st.button("Predict Quality", type="primary"):
                # Make prediction
                input_data = [[depth, rop, wob, torque, rpm, flow_rate, spp]]
                prediction = model.predict(input_data)
                
                st.success(f"Predicted Drilling Quality: **{prediction[0]}**")
    
    elif page == "About":
        st.header("About This Application")
        st.markdown("""
        This application uses machine learning to predict drilling quality based on various drilling parameters.
        
        ### Features:
        - **Data Overview**: View and analyze drilling data
        - **Prediction**: Predict drilling quality based on input parameters
        
        ### Parameters:
        - **Depth**: Current depth of drilling
        - **ROP**: Rate of Penetration
        - **WOB**: Weight on Bit
        - **Torque**: Drilling torque
        - **RPM**: Rotations per minute
        - **Flow Rate**: Mud flow rate
        - **SPP**: Standpipe Pressure
        
        ### Model:
        The model is trained using historical drilling data to classify drilling quality into categories:
        excellent, good, fair, or poor.
        """)

if __name__ == "__main__":
    main()
