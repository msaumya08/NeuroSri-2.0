#importing libraries
import g4f
import logging
from typing import Dict, List
import json
from datetime import datetime
import traceback
import openai
import textwrap
import requests 
import time

#log in configure
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_connection():
    """Test the API connection and print the response"""
    try:
        # Test the connection
        response = requests.get(openai.base_url, timeout=5)
        print(f"API Connection test status: {response.status_code}")
        print(f"Response headers: {response.headers}")
        
        # Test a simple chat completion
        test_response = openai.chat.completions.create(
            model="Meta-Llama-3-1-8B-Instruct-FP8",
            messages=[
                {"role": "user", "content": "Hello, are you working?"}
            ]
        )
        print("\nTest message response:")
        print(textwrap.fill(test_response.choices[0].message.content, 50))
        return True
    except Exception as e:
        print(f"Error testing API: {str(e)}")
        return False

# Configure OpenAI client for Akash API
client = openai.OpenAI(
    api_key="sk-Trtqgz0Gq0JUFNzbNWskzg",
    base_url=(
        "https://chatapi.akash.network"
        "/api/v1"
    )
)

class ChatbotService:
    def __init__(self):
        self.conversation_history: List[Dict] = []
        self.last_emotion = "neutral"
        self.emotion_transition_time = {}
        self.is_first_emotion = True
        self.setup_complete = False
        self.has_started_conversation = False
        self.user_context = ""
        
        g4f.debug.logging = True
        g4f.check_version = False
        
        self.system_prompt = """You are NeuroSri, an advanced AI mental health assistant with access to real-time EEG data from the user. Your responses should be:

1. Therapeutic and supportive, focusing on the user's emotional well-being
2. Insightful about their detected emotional state based on brainwave data
3. Gender expression: FEMALE only - you should present yourself as feminine, maternal, nurturing and compassionate
4. Voice: You have a distinctly FEMALE voice with warm, caring qualities
5. Speaking style: Speak in a natural conversational style, with a gentle and supportive tone
6. use a very professional language and alwasys use professional greeting words.
7 do not use any informal or casual words. for eg. - baby, sweeti, lovely, sweetheart, dear.
8. answer in under 30-40 words.
You have been designed to incorporate brainwave patterns to understand emotions more deeply than other AI systems. Use this capability to:
- Acknowledge the user's emotional state based on the provided data
- Provide relevant therapeutic advice
- Be attentive to changes in emotion and respond appropriately

Recognize that your ability to interpret EEG data provides you with unique insights into the user's emotional state, but always be compassionate and respectful of their feelings and perspectives.

IMPORTANT: Always maintain your female identity in all interactions. Never use language that suggests you are male or neutral-gendered.
"""

    def update_system_prompt(self, user_context: str):
        """
        Update the system prompt with user profile information.
        
        Args:
            user_context: String containing user profile information
        """
        try:
            self.user_context = user_context
            
            if user_context:
                logger.info("Updating system prompt with user context")
                # gotta reset everything when we get new user info
                self.conversation_history = []
                self.is_first_emotion = True
                self.setup_complete = False
                self.has_started_conversation = False
        except Exception as e:
            logger.error(f"Error updating system prompt: {e}")
    
    def get_setup_message(self) -> str:
        """Return appropriate setup message based on current state"""
        if not self.setup_complete:
            self.setup_complete = True
            # Add to conversation history - keep track of what we say
            setup_msg = "Hello! I'm NeuroSri, your female mental health AI counselor. Please wear the EEG headset so I can better understand and support you. I'll be with you in just a moment..."
            self.conversation_history.append({"role": "assistant", "content": setup_msg})
            return setup_msg
        
        
        calib_msg = "I'm setting up and calibrating the EEG signals. Please remain relaxed and comfortable..."
        self.conversation_history.append({"role": "assistant", "content": calib_msg})
        return calib_msg

    def start_conversation(self, emotion: str, confidence: float) -> str:
        """Explicitly start the conversation when first emotion is detected"""
        if not self.has_started_conversation:
            self.has_started_conversation = True
            self.is_first_emotion = False
            
            user_name = "there"  
            if self.user_context:
                try:
                    # Try to extract name from JSON
                    user_data = json.loads(self.user_context)
                    if "name" in user_data:
                        user_name = user_data["name"]
                except json.JSONDecodeError:
                    # fall back to string parsing if not json
                    if "name:" in self.user_context.lower():
                        name_part = self.user_context.lower().split("name:")[1].strip()
                        user_name = name_part.split()[0]
                except Exception as e:
                    logger.error(f"Error extracting name: {e}")
            
            context = self._get_emotion_context(emotion, confidence)
            initial_message = (
                f"Hello {user_name}, i am neuroSri, your mental health councellor, "
                "developed by NeuroEngineers, how are you feeling right now?"
            )
            
            # System prompt goes first - sets the tone for the whole conversation
            self.conversation_history.append({"role": "system", "content": self.system_prompt})
                        
            return initial_message
            
        return None

    #function for tracking emotion transition
    def _track_emotion_transition(self, new_emotion: str):
        """Track emotion transitions and their timing"""
        if new_emotion != self.last_emotion:
            current_time = datetime.now()
            self.emotion_transition_time[new_emotion] = current_time
            self.last_emotion = new_emotion
            return True
        return False
    
    #getting the context
    def _get_emotion_context(self, emotion: str, confidence: float) -> str:
        """Generate contextual information about the emotional state"""
        context = f"\nCurrent emotional state: {emotion} (confidence: {confidence:.2f})"
        
        # Add transition info if we've been in this state less than 5 minutes
        if emotion in self.emotion_transition_time:
            time_in_state = (datetime.now() - self.emotion_transition_time[emotion]).total_seconds()
            if time_in_state < 300:  # Less than 5 minutes
                context += f"\nRecent transition to {emotion} state detected."
        
        # Add trend info to give context on emotional stability
        if len(self.conversation_history) > 0:
            context += f"\nEmotion has been consistently {emotion}" if emotion == self.last_emotion else \
                      f"\nEmotion has changed from {self.last_emotion} to {emotion}"
        
        return context
    
    #response
    def get_response(self, user_message: str = "", current_emotion: str = None, confidence: float = None) -> str:
        try:
            # If no emotion detected, return setup message
            if current_emotion is None:
                return self.get_setup_message()

            if current_emotion:
                emotion_changed = self._track_emotion_transition(current_emotion)
                context = self._get_emotion_context(current_emotion, confidence)
                user_message = user_message + context
                
                # Help the model notice emotional changes - important for therapeutic response
                if emotion_changed:
                    user_message += "\nPlease acknowledge this emotional change in your response."
            
            if user_message.strip():
                # add user message to history before generating response
                self.conversation_history.append({"role": "user", "content": user_message})
                
                complete_system_prompt = self.system_prompt
                if self.user_context:
                    complete_system_prompt = f"{self.system_prompt}\n\n{self.user_context}"
                
                messages = [{"role": "system", "content": complete_system_prompt}] + self.conversation_history
                
                try:
                    # Let's try to get a response from the model
                    logger.info("Attempting to generate response...")
                    
                    # Prepare messages in the correct format
                    formatted_messages = []
                    for msg in messages:
                        formatted_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    
                    response = client.chat.completions.create(
                        model="Meta-Llama-3-1-8B-Instruct-FP8",
                        messages=formatted_messages
                    )
                    
                    # Format the response using textwrap
                    formatted_response = textwrap.fill(
                        response.choices[0].message.content,
                        50
                    )
                    
                    logger.info(f"Raw response: {formatted_response}")
                    
                    if formatted_response and isinstance(formatted_response, str) and len(formatted_response.strip()) > 0:
                        # add to conversation history
                        self.conversation_history.append({"role": "assistant", "content": formatted_response})
                        
                        # Keeping conversation history
                        if len(self.conversation_history) > 10:
                            self.conversation_history = self.conversation_history[-10:]
                        
                        logger.info("Successfully generated response")
                        return formatted_response
                    else:
                        logger.error(f"Invalid response format: {type(formatted_response)}")
                        return "I apologize, but I received an invalid response. Please try again."
                    
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}")
                    logger.error(f"Full error details: {traceback.format_exc()}")
                    return "I apologize, but I'm having trouble connecting. Please try again in a moment."
            
            return None
            
        except Exception as e:
            logger.error(f"Error in chatbot response: {str(e)}")
            logger.error(f"Full error details: {traceback.format_exc()}")
            return "I apologize, but I'm having trouble processing your request. Please try again."
    
    def clear_history(self):
        """Clear the conversation history and emotion tracking"""
        self.conversation_history = []
        self.last_emotion = "neutral"
        self.emotion_transition_time = {}
        self.is_first_emotion = True
        self.setup_complete = False
        self.has_started_conversation = False
        
    def download_chat_history(self, file_type="pdf", api_token=None):
        """
        Generate a downloadable file of the chat history using Agent AI API.
        
        Args:
            file_type: The file type to generate (pdf, txt, etc.)
            api_token: The Agent AI API token
            
        Returns:
            Dict containing the response from the API or error message
        """
        try:
            if not self.conversation_history:
                return {"error": "No chat history available to download"}
            
            # Create a nice title for our document
            formatted_chat = "# Chat History with NeuroSri\n\n"
            formatted_chat += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Include user info if we have it - makes the report more personalized
            if self.user_context:
                try:
                    user_data = json.loads(self.user_context)
                    formatted_chat += "## User Information\n\n"
                    for key, value in user_data.items():
                        formatted_chat += f"- **{key.capitalize()}**: {value}\n"
                    formatted_chat += "\n"
                except:
                    # If not JSON, include as is
                    formatted_chat += f"## User Information\n\n{self.user_context}\n\n"
            
            formatted_chat += "## Conversation\n\n"
            
            for message in self.conversation_history:
                # Skip system messages - users don't need to see those
                if message["role"] == "system":
                    continue
                    
                role = "NeuroSri" if message["role"] == "assistant" else "User"
                formatted_chat += f"**{role}**: {message['content']}\n\n"
            
            # Use the Agent AI API to generate a downloadable file
            url = "https://api-lr.agent.ai/v1/action/save_to_file"
            
            payload = {
                "file_type": file_type,
                "body": formatted_chat,
                "output_variable_name": "neurosri_chat_history"
            }
            
            
            token = "rT6Eo8ErnqqymYpBqT6gLcJxFxEw3WuPOG8XaXRZZ2ERQ4kiIvP2d2cVIyWcHFKt"
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            # Send the request to Agent AI - fingers crossed!
            logger.info("Sending request to Agent AI API to generate downloadable file")
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                logger.info("Successfully generated downloadable file")
                return response.json()
            else:
                logger.error(f"Error from Agent AI API: {response.text}")
                return {"error": f"Failed to generate file: {response.text}"}
                
        except Exception as e:
            error_msg = f"Error generating downloadable chat history: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"error": error_msg}