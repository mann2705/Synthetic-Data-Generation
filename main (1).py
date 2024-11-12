import streamlit as st
import pandas as pd
from io import BytesIO
from data_generation import create_healthcare_knowledge_graph, generate_synthetic_data
from utils import prepare_data_for_training, train_hierarchical_vaegan
from models import HierarchicalVAEGAN
import plotly.express as px
import threading

def main():
    st.set_page_config(page_title="ISD-GRE Healthcare", layout="wide")
    st.title("🧬 Intelligent Synthetic Data Generation for Rare Healthcare Events")

    st.write("### Step-by-Step Guide for Data Generation")
    st.write("""
    1. Select the number of samples you want to generate.
    2. Choose one or more diseases from the list of rare diseases.
    3. Choose one or more genetic variants linked to these diseases.
    4. Select a template if you'd like a predefined setup.
    5. Click 'Generate Synthetic Data' to create the dataset.
    6. After generation, preview the data and download it in your preferred format.
    7. Optionally, train a machine learning model on the generated data.
    """)

    st.sidebar.header("Data Generation Parameters")
    num_samples = st.sidebar.number_input("Number of Samples", min_value=100, max_value=10000, value=1000, step=100)
    disease_options = ['Cystic Fibrosis', 'Hemophilia', 'Huntington\'s Disease']
    genetic_variants_options = ['Mutation X', 'Mutation Y', 'Mutation Z']
    disease_select = st.sidebar.multiselect("Select Rare Diseases", disease_options)
    genetic_variants = st.sidebar.multiselect("Select Genetic Variants", genetic_variants_options)

    # Templates
    st.sidebar.header("Templates")
    template = st.sidebar.selectbox("Choose a Template", ['Custom', 'Rare Genetic Disorder', 'Neurological Study'])

    if template == 'Rare Genetic Disorder':
        disease_select = ['Cystic Fibrosis']
        genetic_variants = ['Mutation X']
        st.sidebar.info("Template 'Rare Genetic Disorder' selected: Pre-filled disease and genetic variant.")

    elif template == 'Neurological Study':
        disease_select = ['Huntington\'s Disease']
        genetic_variants = ['Mutation Y', 'Mutation Z']
        st.sidebar.info("Template 'Neurological Study' selected: Pre-filled disease and genetic variants.")

    if st.sidebar.button("Generate Synthetic Data"):
        if not disease_select or not genetic_variants:
            st.error("Please select at least one disease and one genetic variant.")
        else:
            with st.spinner('Generating synthetic data...'):
                G = create_healthcare_knowledge_graph(disease_select, genetic_variants)
                synthetic_data = generate_synthetic_data(num_samples, G)
            st.success(f"Generated {len(synthetic_data)} synthetic data samples successfully!")

            # Display the number of samples generated
            st.write(f"### Generated {len(synthetic_data)} Samples")

            # Display the data
            if len(synthetic_data) <= 10000:
                st.write("### Preview of the Synthetic Data")
                st.dataframe(synthetic_data)
            else:
                st.write("### Preview of the Synthetic Data (First 5 Samples)")
                st.dataframe(synthetic_data.head())

            # Data Visualization
            st.write("### Data Visualization")
            fig = px.histogram(synthetic_data, x='Disease', color='Gender', barmode='group')
            st.plotly_chart(fig, use_container_width=True)

            # Download options
            st.write("### Download Synthetic Data")
            file_format = st.selectbox("Select File Format", ['CSV', 'Excel'])
            include_data_dict = st.checkbox("Include Data Dictionary")
            if st.button("Download Data"):
                download_synthetic_data(synthetic_data, file_format, include_data_dict)

            # Model training (Optional)
            st.write("### Train a Machine Learning Model (Optional)")
            if st.button("Train Model"):
                with st.spinner('Training model...'):
                    train_model(synthetic_data)
                st.success("Model trained successfully!")

def download_synthetic_data(synthetic_data, file_format, include_data_dict):
    data_dict = pd.DataFrame({
        'Column': synthetic_data.columns,
        'Description': [
            'Unique patient identifier',
            'Age of the patient',
            'Gender of the patient',
            'Genetic variant present',
            'Disease risk level',
            'Disease assigned',
            'Risk score calculated',
            'Simulated lab result 1',
            'Simulated lab result 2',
        ]
    })

    if file_format == 'CSV':
        if include_data_dict:
            # Combine synthetic data and data dictionary
            combined_csv = synthetic_data.to_csv(index=False) + "\n\n" + data_dict.to_csv(index=False)
            csv_data = combined_csv.encode('utf-8')
        else:
            csv_data = synthetic_data.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Data as CSV",
            data=csv_data,
            file_name="synthetic_healthcare_data.csv",
            mime="text/csv"
        )

    elif file_format == 'Excel':
        towrite = BytesIO()
        with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
            synthetic_data.to_excel(writer, index=False, sheet_name='Data')
            if include_data_dict:
                data_dict.to_excel(writer, index=False, sheet_name='Data Dictionary')
            writer.save()
        towrite.seek(0)
        st.download_button(
            label="Download Data as Excel",
            data=towrite,
            file_name="synthetic_healthcare_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def train_model(synthetic_data):
    # Prepare data
    dataset = prepare_data_for_training(synthetic_data)
    input_dims = {
        'genetic': dataset.element_spec['genetic'].shape[-1],
        'clinical': dataset.element_spec['clinical'].shape[-1],
        'environmental': dataset.element_spec['environmental'].shape[-1],
    }
    latent_dim = 10

    # Initialize and train model
    model = HierarchicalVAEGAN(input_dims, latent_dim)
    train_thread = threading.Thread(target=train_hierarchical_vaegan, args=(model, dataset, 5))
    train_thread.start()
    train_thread.join()

if __name__ == "__main__":
    main()
