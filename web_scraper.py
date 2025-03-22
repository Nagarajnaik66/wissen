import os
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any
import time
import requests
from bs4 import BeautifulSoup
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, serpapi_key: str, gemini_api_key: str):
        self.serpapi_key = serpapi_key
        self.gemini_api_key = gemini_api_key
        self.search = SerpAPIWrapper(serpapi_api_key=self.serpapi_key)
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.gemini_api_key,
            temperature=0.3
        )
        
    def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web for a given query and return the top results."""
        try:
            logger.info(f"Searching for: {query}")
            
            # SerpAPIWrapper.results() doesn't accept num_results, so we'll use the run method
            # which returns a string and then handle the parsing ourselves
            raw_results = self.search.run(query)
            
            # Parse the search results manually
            search_params = {
                "engine": "google",
                "q": query,
                "api_key": self.serpapi_key,
                "num": num_results  # Number of results
            }
            
            import json
            from serpapi import GoogleSearch
            
            search = GoogleSearch(search_params)
            results = search.get_dict()
            
            if not results or "organic_results" not in results:
                logger.warning(f"No search results found for: {query}")
                return []
                
            # Extract relevant information from search results
            formatted_results = []
            for result in results["organic_results"][:num_results]:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "link": result.get("link", "")
                })
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching web: {str(e)}")
            return []
    
    def fetch_article_content(self, url: str) -> str:
        """Fetch and parse the content of an article from a given URL."""
        try:
            logger.info(f"Fetching content from: {url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Remove blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Truncate to reasonable length (first 8000 chars)
            text = text[:8000]
            
            return text
        except Exception as e:
            logger.error(f"Error fetching article content: {str(e)}")
            return ""
    
    def analyze_topic(self, topic: str) -> Dict[str, Any]:
        """
        Analyze a topic by searching the web and extracting information.
        Returns a dictionary with the topic information.
        """
        logger.info(f"Analyzing topic: {topic}")
        
        # Search the web for the topic
        search_results = self.search_web(topic)
        
        # Extract content from each result
        content_data = []
        for result in search_results[:3]:  # Limit to top 3 results to avoid rate limiting
            content = self.fetch_article_content(result["link"])
            if content:
                content_data.append({
                    "title": result["title"],
                    "content": content,
                    "url": result["link"]
                })
                # Sleep to avoid rate limiting
                time.sleep(1)
        
        # Combine all content
        combined_content = "\n\n".join([
            f"SOURCE: {item['title']}\n{item['content']}" for item in content_data
        ])
        
        # Use LLM to summarize and structure the content
        summary = self._summarize_content(topic, combined_content)
        
        return {
            "topic": topic,
            "summary": summary,
            "sources": [item["url"] for item in content_data]
        }
    
    def _summarize_content(self, topic: str, content: str) -> str:
        """Use the LLM to summarize and structure the content."""
        try:
            prompt_template = PromptTemplate.from_template(
                """You are a researcher tasked with summarizing information about {topic}.
                The following is content collected from various web sources:
                
                {content}
                
                Please provide a comprehensive summary of the topic in about 500 words.
                Focus on the most important aspects and ensure the information is accurate.
                Do not include any opinions or biases.
                """
            )
            
            prompt = prompt_template.format(topic=topic, content=content)
            response = self.model.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error summarizing content: {str(e)}")
            return "Failed to summarize content due to an error."