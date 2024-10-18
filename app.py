
import streamlit as st
import pandas as pd
import torch
from transformers import TapexTokenizer, BartForConditionalGeneration

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
    model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")
    return tokenizer, model

# Function to generate the answer based on the query
def generate_answer(table, query, tokenizer, model):
    encoding = tokenizer(table=table, query=query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**encoding)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# Main app code
def main():
    st.title("Retail Sales Chatbot")

    # Load model
    tokenizer, model = load_model()

    # User file upload
    uploaded_file = st.file_uploader("Upload your retail sales CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())

        # Extract the columns from the uploaded CSV
        uploaded_columns = df.columns.tolist()
        st.write("Uploaded Columns:", uploaded_columns)

        # Prepare the data directly from the uploaded DataFrame
        data_dict = {col: df[col].tolist() for col in uploaded_columns if col in df.columns}

        # Convert to DataFrame
        table = pd.DataFrame(data_dict)

        # User input for the query
        user_query = st.text_input("Ask a question about the retail sales data:")

        if st.button("Get Answer"):
            if user_query:
                answer = generate_answer(table, user_query, tokenizer, model)
                st.write("**Answer:**", answer)
            else:
                st.warning("Please enter a query.")

    else:
        st.info("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()
