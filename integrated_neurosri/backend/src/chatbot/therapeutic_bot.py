from g4f.client import Client
import sys
from pathlib import Path
import random
import logging
from typing import Dict, Any, Optional
import traceback
import requests

from .chatbot_service import ChatbotService

# gotta add the project root to path or imports will break 🙄
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.config.settings import CHATBOT_SYSTEM_PROMPT, MAX_RESPONSE_LENGTH

logger = logging.getLogger(__name__)

class TherapeuticBot:
    def __init__(self):
        self.conversation_history = []

        # setup system prompt - we NEED to make sure it sounds female!
        if not hasattr(self, 'system_prompt') or self.system_prompt is None:
            self.system_prompt = CHATBOT_SYSTEM_PROMPT
            if "female" not in self.system_prompt.lower():
                self.system_prompt = self.system_prompt.replace(
                    "You are NeuroSri,", 
                    "You are NeuroSri, a female AI counselor with a nurturing, feminine voice,"
                )
        self.last_emotion = None
        self.client = Client()
        self.chatbot_service = ChatbotService()
        self.user_profile = None
        logger.info("TherapeuticBot initialized")
        
    def update_user_profile(self, user_info: Dict[str, Any]) -> None:
        """
        Updates the user profile with form data - helps make responses personal!
        
        Args:
            user_info: Dict with all the juicy user details
        """
        try:
            self.user_profile = user_info
            logger.info(f"User profile updated: {user_info['name']}, age: {user_info['age']}")
            
            # Create and update context - this is what makes responses feel "magical"
            user_context = self._create_user_context_prompt()
            self.chatbot_service.update_system_prompt(user_context)
            
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            logger.error(traceback.format_exc())
    
    def _create_user_context_prompt(self) -> str:
        """
        Builds a nice personalized prompt for our bot - makes it feel like it "knows" the user.
        
        Returns:
            A fancy formatted string with all the user info (if we have any)
        """
        if not self.user_profile:
            return ""
        
        try:
            profile = self.user_profile
            
            # build a nice formatted user profile - makes the AI responses way better!
            user_context = (
                f"User Profile Information:\n"
                f"- Name: {profile.get('name', 'Unknown')}\n"
                f"- Age: {profile.get('age', 'Unknown')}\n"
                f"- Gender: {profile.get('gender', 'Unknown')}\n"
                f"- Main concern: {profile.get('mainConcern', 'Not specified')}\n"
            )
            
            # add the extra stuff if we have it - more is better, right?
            if profile.get('previousTherapy'):
                user_context += f"- Previous therapy experience: {profile.get('previousTherapy')}\n"
            
            if profile.get('sleepQuality'):
                user_context += f"- Sleep quality: {profile.get('sleepQuality')}\n"
                
            if profile.get('stressLevel'):
                user_context += f"- Stress level: {profile.get('stressLevel')}\n"
            
            # IMPORTANT: tell the bot how to use this info without being creepy
            user_context += (
                "\nUse this information to personalize your responses. "
                "Refer to the user by name and tailor your therapeutic approach "
                "based on their specific concerns and background. "
                "However, avoid directly mentioning that you have this information "
                "unless it's relevant to the conversation flow."
            )
            
            return user_context
            
        except Exception as e:
            logger.error(f"Error creating user context prompt: {e}")
            return ""  # meh, just return empty string if it fails
    
    def generate_response(self, user_input: str, emotion: str = "neutral", 
                         confidence: float = 0.0, user_info: Optional[Dict[str, Any]] = None) -> str:
        """
        The main function that makes the bot talk! Give it user text, get AI response.
        
        Args:
            user_input: What the human said
            emotion: How they're feeling (from EEG)
            confidence: How sure we are about the emotion (0-1)
            user_info: Optional user profile stuff
            
        Returns:
            Whatever the bot decides to say back
        """
        try:
            # Update profile if we got new info - but only once! 
            if user_info and not self.user_profile:
                self.update_user_profile(user_info)
            
            # Add logging to track the flow
            logger.info(f"Generating response for input: {user_input}")
            logger.info(f"Current emotion: {emotion}, confidence: {confidence}")
            
            # This is where the magic happens... *fingers crossed*
            response = self.chatbot_service.get_response(user_input, emotion, confidence)
            
            if not response:
                logger.warning("Empty response from chatbot service")
                return "I'm sorry, I couldn't generate a response. Could you please try again?"
            
            return response
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return "I'm having trouble connecting to my language model. Please make sure Ollama is running."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            return "I apologize, but I'm having trouble processing your request right now. Please try again in a moment."
            
    def _get_brain_activity_description(self, emotion: str) -> str:
        """Turns technical EEG stuff into words normal people understand"""
        # people LOVE when we talk about their brain waves! makes it feel scientific
        if emotion.lower() == 'relaxed':
            return "Showing increased alpha waves, indicating a calm and focused state"
        elif emotion.lower() == 'stressed':
            return "Showing elevated beta activity, suggesting heightened alertness or tension"
        else:
            return "Showing typical patterns within normal ranges"  # idk, just say something neutral
            
    def _get_fallback_response(self, emotion: str) -> str:
        """If all else fails, grab one of these pre-written responses"""
        # TODO: add more responses for different emotions - these are getting stale
        fallback_responses = {
            'stressed': [
                "Hey there! I notice your brainwaves are indicating some stress. Would you like to try a quick breathing exercise together?",
                "I can see from your EEG patterns that you might be feeling under pressure. Let's take a moment to focus on what might help you feel more at ease.",
                "Your brainwave activity suggests you're experiencing stress. Would you like to explore some simple relaxation techniques I know?"
            ],
            'relaxed': [
                "I'm really glad to see your brainwave patterns showing a relaxed state! Would you like to explore what's contributing to this positive state?",
                "Your EEG readings show a wonderfully calm pattern. What would help you maintain this peaceful state?",
                "I'm seeing some great alpha wave activity indicating relaxation. Would you like to discuss ways to preserve this feeling?"
            ],
            'default': [
                "Hey! I'm NeuroSri, your female AI companion, and I'm here to support you. Would you like to explore what you're feeling right now?",
                "I'm noticing some interesting patterns in your brainwaves. Could you tell me more about what's on your mind?",
                "Let's focus on what would be most helpful for you right now. What would you like to discuss?"
            ]
        }
        
        # pick a random one so it's not boring - nobody likes repetitive bots!
        responses = fallback_responses.get(emotion.lower(), fallback_responses['default'])
        return random.choice(responses)
            
    def _enhance_system_prompt(self, emotion: str) -> str:
        """Makes the system prompt match the user's current emotion - super important!"""

        base_prompt = self.system_prompt
        # FIXME: sometimes female identity isn't coming through clearly enough
        if "female" not in base_prompt.lower():
            base_prompt = base_prompt.replace(
                "You are NeuroSri,", 
                "You are NeuroSri, a female AI counselor with a nurturing, feminine voice,"
            )
            
        # different emotions need different approaches - this is the secret sauce!
        emotion_specific_guidance = {
            'stressed': "Focus on calming techniques and stress relief strategies. Use a gentle, reassuring, nurturing feminine tone. Provide specific relaxation exercises.",
            'sad': "Offer emotional support and validation with a warm, compassionate feminine voice. Help explore and process feelings with empathy. Suggest mood-lifting activities when appropriate.",
            'angry': "Acknowledge feelings while helping to process anger constructively. Maintain a calm, steady, nurturing presence. Offer anger management techniques with a soothing feminine voice.",
            'happy': "Reinforce positive emotions and explore what's contributing to the good mood with an enthusiastic, warm feminine tone. Help build on positive experiences.",
            'relaxed': "Maintain the calm state while exploring positive aspects of their situation with a soft, gentle feminine voice. Encourage mindfulness and present-moment awareness.",
            'neutral': "Focus on open exploration and general well-being with a warm, inviting feminine tone. Help identify any underlying emotions or thoughts."
        }
        
        # glue it all together - this makes a HUGE difference in quality!
        enhanced_prompt = (
            base_prompt + 
            "\n\nCurrent emotional context: " + 
            emotion_specific_guidance.get(emotion.lower(), "Provide balanced, supportive responses with a nurturing feminine voice.") +
            "\n\nRemember to:\n" +
            "1. Be conversational and natural with a distinctly feminine voice\n" +
            "2. Ask open-ended questions in a nurturing manner\n" +
            "3. Validate emotions before offering suggestions\n" +
            "4. Keep responses concise but meaningful\n" +
            "5. Always maintain your feminine identity and voice"
        )
        return enhanced_prompt
            
    def clear_history(self):
        """Wipes the conversation clean - good for testing or starting over"""
        # sometimes you just need a fresh start, ya know?
        self.conversation_history = []
        self.last_emotion = None
        
    def get_conversation_history(self) -> list:
        """Just returns the current conversation - handy for saving or analyzing"""
        return self.conversation_history 