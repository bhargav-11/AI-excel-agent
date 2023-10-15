import streamlit as st
import csv

import concurrent.futures
from langchain.utilities import SerpAPIWrapper
import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

from langchain.tools import Tool
from langchain.utilities import GoogleSerperAPIWrapper
import time
import base64
import pandas as pd
import io


prompt = """
You are a helpful assistant to validate information.
Based on below provided 'Company' and 'Designation' if the Company is in B2C(Business to Consumer) and 
if the Designation is in Marketing at a high level return 'True' else return 'False'.

First check if the Company is B2C(Business to Consumer), If it is then only move to second condition which is 
checking if the Designation is in Marketing at a high level. If the Company is not in B2C(Business to Consumer)
then directly return 'False', without moving to second step.

------------------------------------------------------------
Company: {}
Designation: {}
"""

from langchain.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Google Search",
        func=search.run,
        description="Search Google for recent results.",
    )
]

output_file_path = 'result.csv'

def process_row(agent, row):
    try:
        # print(row)
        result = agent.run(prompt.format(row[2], row[3]))
        row.append(result)
    except Exception as e:
        st.write(e)
        row.append("ERROR")
    
    return row

def write_row(row):
    with open(output_file_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(row)

def process_file(agent, uploaded_file):
    processed_rows = []
    
    # Reading CSV with pandas
    df = pd.read_csv(uploaded_file)
    counter = 0
    
    for index, row in df.iterrows():
        processed_row = process_row(agent, row.tolist())  # Converting row to list
        write_row(processed_row)
        processed_rows.append(processed_row)
        counter += 1
        if counter % 30 == 0:
            st.write("---------------> processed {} rows".format(counter))
            # Convert to CSV string while properly handling commas
            csv_str = convert_to_csv_str(processed_rows)

            # Encode CSV string in Base64
            b64 = base64.b64encode(csv_str.encode()).decode()

            # Create a download link in Streamlit
            st.markdown(f'<a href="data:file/csv;base64,{b64}" download="result.csv">Download processed CSV</a>', unsafe_allow_html=True)

    return processed_rows

def get_csv_download_link(csv_data, filename="result.csv", link_text="Download processed CSV"):
    import base64
    csv_str = "\n".join([",".join(map(str, row)) for row in csv_data])
    b64 = base64.b64encode(csv_str.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'

def convert_to_csv_str(data):
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    writer.writerows(data)
    return output.getvalue()

def authenticate():
    # Replace 'your_password' with the actual password you want to use
    password = st.text_input("Enter password:", type="password")
    return password == 'AIalltheway@2166'

def main():
    st.title("AI Data Sorting")

    if authenticate():
        api_key = st.text_input("Enter API Key:")

        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file:
            if st.button("Process"):
                if not api_key:
                    st.warning("Please enter an API key!")
                else:
                    # Create the output file or clear it if it exists
                    with open(output_file_path, 'w', newline='') as file:
                        pass

                    llm = ChatOpenAI(temperature=0.5, openai_api_key=api_key, model="gpt-4-0613")
                    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
                    result = process_file(agent, uploaded_file)
                    st.markdown(get_csv_download_link(result), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
