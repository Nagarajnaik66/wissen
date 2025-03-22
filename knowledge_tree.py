from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeTreeManager:
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.gemini_api_key,
            temperature=0.2
        )
    
    def generate_knowledge_tree(self, topic: str, content: str) -> Dict[str, Any]:
        """
        Generate a knowledge tree for a given topic and content.
        Returns a dictionary representing the knowledge tree.
        """
        logger.info(f"Generating knowledge tree for: {topic}")
        
        # Use the LLM to generate a knowledge tree
        prompt_template = PromptTemplate.from_template(
            """You are a knowledge organizer tasked with creating a structured knowledge tree about {topic}.
            
            Based on the following content:
            
            {content}
            
            Generate a knowledge tree with the following structure:
            1. Main topic (the topic provided)
            2. 3-5 major subtopics
            3. For each subtopic, 3-5 key points or concepts
            4. For each key point, a brief explanation (1-2 sentences)
            
            Format your response as a JSON object with the following structure:
            {{
                "topic": "Main Topic",
                "subtopics": [
                    {{
                        "name": "Subtopic 1",
                        "key_points": [
                            {{
                                "point": "Key Point 1",
                                "explanation": "Brief explanation of Key Point 1"
                            }},
                            // more key points...
                        ]
                    }},
                    // more subtopics...
                ]
            }}
            
            Ensure the structure is comprehensive and covers the most important aspects of the topic.
            """
        )
        
        try:
            prompt = prompt_template.format(topic=topic, content=content)
            response = self.model.invoke(prompt)
            
            # Extract JSON from the response
            json_str = self._extract_json(response.content)
            knowledge_tree = json.loads(json_str)
            
            return knowledge_tree
        except Exception as e:
            logger.error(f"Error generating knowledge tree: {str(e)}")
            # Return a basic structure if there's an error
            return {
                "topic": topic,
                "subtopics": [
                    {
                        "name": "Error generating knowledge tree",
                        "key_points": [
                            {
                                "point": "Error",
                                "explanation": f"Failed to generate knowledge tree: {str(e)}"
                            }
                        ]
                    }
                ]
            }
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain markdown code blocks or other text."""
        # Look for JSON between triple backticks
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        
        if json_match:
            return json_match.group(1)
        
        # If no code blocks, try to find JSON between curly braces
        json_match = re.search(r'({[\s\S]*})', text)
        if json_match:
            return json_match.group(1)
        
        # If nothing found, return the original text
        return text
    
    def research_topic(self, topic: str, summary: str, sources: List[str]) -> Dict[str, Any]:
        """
        Research a topic and generate a knowledge tree.
        Returns a dictionary with the topic information and knowledge tree.
        """
        logger.info(f"Researching topic: {topic}")
        
        # Generate a knowledge tree based on the summary
        knowledge_tree = self.generate_knowledge_tree(topic, summary)
        
        # Add sources to the knowledge tree
        knowledge_tree["sources"] = sources
        
        return knowledge_tree
    
    def expand_subtopic(self, topic: str, subtopic: str, content: str) -> Dict[str, Any]:
        """
        Expand a subtopic with more detailed information.
        Returns a dictionary with the expanded subtopic information.
        """
        logger.info(f"Expanding subtopic: {subtopic} for topic: {topic}")
        
        prompt_template = PromptTemplate.from_template(
            """You are a knowledge organizer tasked with expanding detailed information about the subtopic {subtopic} 
            within the main topic {topic}.
            
            Based on the following content:
            
            {content}
            
            Generate a detailed expansion of this subtopic with the following structure:
            1. Brief overview of the subtopic (2-3 sentences)
            2. 3-5 key aspects or components of this subtopic
            3. For each aspect, provide detailed information (2-3 paragraphs)
            4. Include any relevant examples, case studies, or applications
            
            Format your response as a JSON object with the following structure:
            {{
                "subtopic": "{subtopic}",
                "overview": "Brief overview text",
                "aspects": [
                    {{
                        "name": "Aspect 1",
                        "details": "Detailed information about Aspect 1",
                        "examples": ["Example 1", "Example 2"]
                    }},
                    // more aspects...
                ]
            }}
            
            Ensure the information is accurate, comprehensive, and well-structured.
            """
        )
        
        try:
            prompt = prompt_template.format(topic=topic, subtopic=subtopic, content=content)
            response = self.model.invoke(prompt)
            
            # Extract JSON from the response
            json_str = self._extract_json(response.content)
            expanded_subtopic = json.loads(json_str)
            
            return expanded_subtopic
        except Exception as e:
            logger.error(f"Error expanding subtopic: {str(e)}")
            return {
                "subtopic": subtopic,
                "overview": f"Failed to expand subtopic: {str(e)}",
                "aspects": []
            }