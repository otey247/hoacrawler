import streamlit as st
from openai import AzureOpenAI
import requests
import pandas as pd
import json
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)

# Import crawl4ai components
from crawl4ai.web_crawler import WebCrawler
from crawl4ai.chunking_strategy import *
from crawl4ai.extraction_strategy import *
from crawl4ai.crawler_strategy import *
from functools import lru_cache

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API credentials
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_KEY")
model = os.getenv("AZURE_OPENAI_DEPLOYMENT")

client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-02-01",
    azure_endpoint=endpoint
)

# Set up Bing Search API credentials
bing_api_key = os.getenv("BING_API_KEY")
bing_endpoint = "https://api.bing.microsoft.com/v7.0/search"

@lru_cache()
def create_crawler():
    crawler = WebCrawler(verbose=False)
    crawler.warmup()
    return crawler

def bing_search(query):
    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
    params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(bing_endpoint, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def crawl_web_pages(urls, max_pages=5):
    """
    Fetches the text content from the given list of URLs using crawl4ai.

    Args:
        urls (list): List of URLs to crawl.
        max_pages (int): Maximum number of pages to crawl.

    Returns:
        dict: Dictionary containing crawled content for each URL.
    """
    crawled_data = {}
    crawler = create_crawler()
    for idx, url in enumerate(urls[:max_pages]):
        try:
            result = crawler.run(url=url, only_text=True)
            text = result.extracted_content
            if text:
                crawled_data[url] = text
            else:
                st.warning(f"No text extracted from {url}.")
        except Exception as e:
            st.warning(f"Failed to crawl {url}: {e}")
            continue
    return crawled_data

def extract_information(search_results, property_name, city, state, zip_code):
    # Get URLs from search results
    urls = [result.get("url") for result in search_results.get("webPages", {}).get("value", [])]
    
    # Crawl the web pages to get their content
    crawled_data = crawl_web_pages(urls)
    
    # Save the crawled data
    query = f"{property_name} {city} {state} {zip_code} current board members contact info"
    crawled_filename = save_crawled_data(crawled_data, query)
    st.success(f"Crawled data saved to {crawled_filename}")
    
    if not crawled_data:
        st.error("No content was crawled from the web pages.")
        return None

    # Prepare the prompt for OpenAI
    crawled_content = "\n\n".join([f"Content from {url}:\n{text}" for url, text in crawled_data.items()])
    prompt = f"""
Extract the current board members along with their contact information (Name, Title/Role, Email, Phone, LinkedIn, Source) from the following text for the property "{property_name}" located in {city}, {state}, {zip_code}.
Text:
{crawled_content}
Provide the information in JSON format as a list of records. If no relevant information is found, return an empty list.
"""

    # Call OpenAI API
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You extract structured data from provided text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0
        )
        extracted_data = response.choices[0].message.content
        logging.info(f"Raw extracted data: {extracted_data}")
        
        # Try to parse the JSON
        try:
            parsed_data = json.loads(extracted_data)
            if not parsed_data:  # If the list is empty
                st.warning("No relevant information found in the crawled data.")
            return json.dumps(parsed_data, indent=2)  # Return prettified JSON string
        except json.JSONDecodeError as json_err:
            st.error(f"Failed to parse JSON: {json_err}")
            logging.error(f"JSON parsing error: {json_err}")
            logging.error(f"Problematic JSON string: {extracted_data}")
            
            # Attempt to clean and parse the JSON
            cleaned_data = extracted_data.strip()
            if cleaned_data.startswith("```json"):
                cleaned_data = cleaned_data[7:]
            if cleaned_data.endswith("```"):
                cleaned_data = cleaned_data[:-3]
            
            try:
                parsed_data = json.loads(cleaned_data)
                st.warning("JSON was cleaned and successfully parsed.")
                return json.dumps(parsed_data, indent=2)
            except json.JSONDecodeError:
                st.error("Failed to parse JSON even after cleaning.")
                return None
    
    except Exception as e:
        st.error(f"An error occurred during OpenAI API call: {e}")
        logging.error(f"OpenAI API error: {e}")
        return None

def save_search_results(search_results, query):
    # Create a 'search_results' directory if it doesn't exist
    if not os.path.exists('search_results'):
        os.makedirs('search_results')

    # Create a valid filename from the query
    filename = "".join(x for x in query if x.isalnum() or x in [' ', '-', '_']).rstrip()
    filename = filename.replace(' ', '_') + '.json'

    # Save the search results to a JSON file
    with open(os.path.join('search_results', filename), 'w') as f:
        json.dump(search_results, f, indent=2)

    return filename

def save_crawled_data(crawled_data, query):
    # Create a 'crawled_data' directory if it doesn't exist
    if not os.path.exists('crawled_data'):
        os.makedirs('crawled_data')

    # Create a valid filename from the query
    filename = "".join(x for x in query if x.isalnum() or x in [' ', '-', '_']).rstrip()
    filename = filename.replace(' ', '_') + '_crawled_data.json'

    # Save the crawled data to a JSON file
    with open(os.path.join('crawled_data', filename), 'w') as f:
        json.dump(crawled_data, f, indent=2)

    return filename

def main():
    st.title("Board Member Information Finder")
    st.write("Enter the property details to find current board members and their contact information.")

    with st.form(key='search_form'):
        property_name = st.text_input("Property Name")
        city = st.text_input("City")
        state = st.text_input("State (Required)")
        zip_code = st.text_input("Zip Code")
        submit_button = st.form_submit_button(label='Search')

    if submit_button:
        if not state:
            st.error("State is required.")
        else:
            query = f"{property_name} {city} {state} {zip_code} current board members contact info"
            st.write(f"**Searching for:** {query}")

            # Perform Bing Search
            try:
                search_results = bing_search(query)
                st.success("Search completed successfully.")

                # Save search results to JSON file
                filename = save_search_results(search_results, query)
                st.success(f"Search results saved to {filename}")
            except Exception as e:
                st.error(f"An error occurred during the search: {e}")
                return

            # Extract Information using OpenAI
            extracted_data = extract_information(search_results, property_name, city, state, zip_code)
            if extracted_data is None:
                st.error("Failed to extract data. Please check the logs for more information.")
                return
            
            try:
                # Convert the extracted JSON data to a pandas DataFrame
                data_list = json.loads(extracted_data)
                if not data_list:
                    st.warning("No relevant information found.")
                else:
                    st.success("Data extracted successfully.")
                    df = pd.DataFrame(data_list)
                    st.dataframe(df)

                    # Provide option to download CSV
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name='board_members.csv',
                        mime='text/csv',
                    )
            except Exception as e:
                st.error(f"An error occurred while processing the extracted data: {e}")
                st.text("Raw extracted data:")
                st.code(extracted_data)

if __name__ == "__main__":
    main()