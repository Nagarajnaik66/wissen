import streamlit as st
import os
from dotenv import load_dotenv
from web_scraper import WebScraper
from knowledge_tree import KnowledgeTreeManager
import json
import time

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Knowledge Tree Generator",
    page_icon="🌳",
    layout="wide"
)

# Initialize session state variables
if 'knowledge_tree' not in st.session_state:
    st.session_state.knowledge_tree = None
if 'current_subtopic' not in st.session_state:
    st.session_state.current_subtopic = None
if 'expanded_subtopic' not in st.session_state:
    st.session_state.expanded_subtopic = None

# Get API keys from environment variables
serpapi_key = os.getenv("SERP_API_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Validate API keys
if not serpapi_key or not gemini_api_key:
    st.error("API keys not found. Please make sure SERPAPI_API_KEY and GOOGLE_API_KEY are set in your .env file.")
    st.stop()

# Application title and description
st.title("🌳 Knowledge Tree Generator")
st.markdown("""
This application researches any topic on the web and organizes the information into a
comprehensive knowledge tree structure. Enter a topic of interest, and the application will:
1. Search the web for relevant information
2. Extract and analyze the content
3. Generate a structured knowledge tree
4. Allow you to explore subtopics in detail
""")

# Main search interface
st.header("Research a Topic")
topic = st.text_input("Enter a topic to research:", placeholder="e.g., Quantum Computing")
num_results = st.slider("Number of search results to analyze:", min_value=1, max_value=10, value=3)

# Search button
if st.button("Generate Knowledge Tree"):
    if not topic:
        st.error("Please enter a topic to research.")
    else:
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize components
            status_text.text("Initializing web scraper...")
            web_scraper = WebScraper(serpapi_key, gemini_api_key)
            knowledge_tree_manager = KnowledgeTreeManager(gemini_api_key)
            
            # Search the web
            status_text.text("Searching the web for information...")
            progress_bar.progress(20)
            search_results = web_scraper.search_web(topic, num_results=num_results)
            
            if not search_results:
                st.error("No search results found. Please try a different topic.")
                status_text.empty()
                progress_bar.empty()
            else:
                # Extract content from search results
                status_text.text("Extracting content from websites...")
                progress_bar.progress(40)
                content_data = []
                for i, result in enumerate(search_results):
                    content = web_scraper.fetch_article_content(result["link"])
                    if content:
                        content_data.append({
                            "title": result["title"],
                            "content": content,
                            "url": result["link"]
                        })
                    progress_bar.progress(40 + (i+1) * 20 // len(search_results))
                
                # Combine all content
                combined_content = "\n\n".join([
                    f"SOURCE: {item['title']}\n{item['content']}" for item in content_data
                ])
                
                # Generate knowledge tree
                status_text.text("Generating knowledge tree...")
                progress_bar.progress(80)
                knowledge_tree = knowledge_tree_manager.generate_knowledge_tree(topic, combined_content)
                
                # Add sources to the knowledge tree
                knowledge_tree["sources"] = [item["url"] for item in content_data]
                
                # Save to session state
                st.session_state.knowledge_tree = knowledge_tree
                st.session_state.current_subtopic = None
                st.session_state.expanded_subtopic = None
                
                # Complete
                status_text.text("Knowledge tree generated successfully!")
                progress_bar.progress(100)
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
                
                # Force rerun to display the knowledge tree
                st.rerun()
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Display the knowledge tree if available
if st.session_state.knowledge_tree:
    st.header(f"Knowledge Tree: {st.session_state.knowledge_tree['topic']}")
    
    # Create two columns for the layout
    col1, col2 = st.columns([1, 2])
    
    # Display the knowledge tree structure in the first column
    with col1:
        st.subheader("Topic Structure")
        
        # Display each subtopic as a button
        for i, subtopic in enumerate(st.session_state.knowledge_tree.get("subtopics", [])):
            if st.button(f"📚 {subtopic['name']}", key=f"subtopic_{i}"):
                st.session_state.current_subtopic = subtopic
                st.session_state.expanded_subtopic = None
                st.rerun()
    
    # Display the selected subtopic in the second column
    with col2:
        if st.session_state.current_subtopic:
            st.subheader(f"Subtopic: {st.session_state.current_subtopic['name']}")
            
            # Display key points
            for i, point in enumerate(st.session_state.current_subtopic.get("key_points", [])):
                st.markdown(f"**{point['point']}**")
                st.markdown(f"_{point['explanation']}_")
                st.markdown("---")
            
            # Option to expand this subtopic
            if st.button("🔍 Expand this subtopic") and not st.session_state.expanded_subtopic:
                try:
                    status_text = st.empty()
                    status_text.text("Expanding subtopic with more details...")
                    
                    # Initialize components if not already
                    knowledge_tree_manager = KnowledgeTreeManager(gemini_api_key)
                    
                    # Get combined content from the knowledge tree
                    combined_content = json.dumps(st.session_state.knowledge_tree)
                    
                    # Expand the subtopic
                    expanded_subtopic = knowledge_tree_manager.expand_subtopic(
                        st.session_state.knowledge_tree['topic'],
                        st.session_state.current_subtopic['name'],
                        combined_content
                    )
                    
                    # Save to session state
                    st.session_state.expanded_subtopic = expanded_subtopic
                    
                    # Complete
                    status_text.empty()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"An error occurred while expanding the subtopic: {str(e)}")
        
        # Display expanded subtopic if available
        if st.session_state.expanded_subtopic:
            st.subheader(f"Detailed: {st.session_state.expanded_subtopic['subtopic']}")
            st.markdown(f"**Overview:** {st.session_state.expanded_subtopic['overview']}")
            
            for aspect in st.session_state.expanded_subtopic.get("aspects", []):
                with st.expander(f"🔎 {aspect['name']}"):
                    st.markdown(aspect['details'])
                    if aspect.get('examples'):
                        st.markdown("**Examples:**")
                        for example in aspect['examples']:
                            st.markdown(f"- {example}")
    
    # Display sources
    st.header("Sources")
    for i, source in enumerate(st.session_state.knowledge_tree.get("sources", [])):
        st.markdown(f"{i+1}. [{source}]({source})")

# Application footer
st.markdown("---")
st.markdown("Knowledge Tree Generator | Created with LangChain, Google Gemini, and SerpAPI")