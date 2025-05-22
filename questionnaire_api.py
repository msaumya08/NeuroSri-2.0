import json
from flask import Flask, request, jsonify
from disorder_questionnaire import DisorderQuestionnaire
import logging
import threading
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('questionnaire_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Initialize questionnaire handler
questionnaire_handler = DisorderQuestionnaire()

# Active questionnaires (disorder -> session_id)
active_questionnaires = {}
# Session data (session_id -> {disorder, current_question, responses})
sessions = {}

@app.route('/api/questionnaire/available', methods=['GET'])
def get_available_questionnaires():
    """Get list of all available questionnaires"""
    try:
        disorders = questionnaire_handler.get_all_disorders()
        return jsonify({
            "status": "success",
            "questionnaires": disorders
        })
    except Exception as e:
        logger.error(f"Error getting available questionnaires: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to retrieve available questionnaires"
        }), 500

@app.route('/api/questionnaire/start', methods=['POST'])
def start_questionnaire():
    """Start a new questionnaire session"""
    try:
        data = request.json
        disorder = data.get('disorder')
        
        if not disorder:
            return jsonify({
                "status": "error",
                "message": "Disorder name is required"
            }), 400
        
        # Get questionnaire for the disorder
        questionnaire = questionnaire_handler.get_questionnaire(disorder)
        if not questionnaire:
            return jsonify({
                "status": "error",
                "message": f"No questionnaire found for '{disorder}'"
            }), 404
        
        # Generate session ID
        session_id = f"{disorder.replace(' ', '_')}_{int(time.time())}"
        
        # Store session data
        sessions[session_id] = {
            "disorder": disorder,
            "current_question": 0,
            "responses": [],
            "total_questions": len(questionnaire["questions"])
        }
        
        # Remember active questionnaire for this disorder
        active_questionnaires[disorder] = session_id
        
        # Return first question
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "questionnaire_name": questionnaire["name"],
            "introduction": questionnaire["introduction"],
            "current_question": 0,
            "total_questions": len(questionnaire["questions"]),
            "question": questionnaire["questions"][0],
        })
    
    except Exception as e:
        logger.error(f"Error starting questionnaire: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to start questionnaire"
        }), 500

@app.route('/api/questionnaire/answer', methods=['POST'])
def answer_question():
    """Submit an answer and get the next question"""
    try:
        data = request.json
        session_id = data.get('session_id')
        answer = data.get('answer')
        
        if not session_id or answer is None:
            return jsonify({
                "status": "error",
                "message": "Session ID and answer are required"
            }), 400
        
        # Get session data
        session = sessions.get(session_id)
        if not session:
            return jsonify({
                "status": "error",
                "message": "Invalid or expired session"
            }), 404
        
        # Store the response
        session["responses"].append(answer)
        current_question = session["current_question"]
        
        # Move to next question
        current_question += 1
        session["current_question"] = current_question
        
        # Get the questionnaire
        disorder = session["disorder"]
        questionnaire = questionnaire_handler.get_questionnaire(disorder)
        
        # Check if we've reached the end of the questionnaire
        if current_question >= len(questionnaire["questions"]):
            # Score the responses
            result = questionnaire_handler.score_responses(disorder, session["responses"])
            
            # Clear session data
            if session_id in sessions:
                del sessions[session_id]
            if disorder in active_questionnaires and active_questionnaires[disorder] == session_id:
                del active_questionnaires[disorder]
            
            # Return final result
            return jsonify({
                "status": "success",
                "completed": True,
                "result": result
            })
        
        # Return next question
        return jsonify({
            "status": "success",
            "completed": False,
            "session_id": session_id,
            "current_question": current_question,
            "total_questions": session["total_questions"],
            "question": questionnaire["questions"][current_question],
        })
    
    except Exception as e:
        logger.error(f"Error processing answer: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to process answer"
        }), 500

@app.route('/api/questionnaire/session/<session_id>', methods=['GET'])
def get_session_status(session_id):
    """Get the current status of a questionnaire session"""
    try:
        # Get session data
        session = sessions.get(session_id)
        if not session:
            return jsonify({
                "status": "error",
                "message": "Invalid or expired session"
            }), 404
        
        # Get the questionnaire
        disorder = session["disorder"]
        questionnaire = questionnaire_handler.get_questionnaire(disorder)
        current_question = session["current_question"]
        
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "disorder": disorder,
            "questionnaire_name": questionnaire["name"],
            "current_question": current_question,
            "total_questions": session["total_questions"],
            "question": questionnaire["questions"][current_question] if current_question < len(questionnaire["questions"]) else None,
            "responses_count": len(session["responses"])
        })
    
    except Exception as e:
        logger.error(f"Error getting session status: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to get session status"
        }), 500

@app.route('/api/questionnaire/cancel/<session_id>', methods=['POST'])
def cancel_session(session_id):
    """Cancel a questionnaire session"""
    try:
        # Get session data
        session = sessions.get(session_id)
        if not session:
            return jsonify({
                "status": "error",
                "message": "Invalid or expired session"
            }), 404
        
        # Get the disorder
        disorder = session["disorder"]
        
        # Remove session data
        if session_id in sessions:
            del sessions[session_id]
        if disorder in active_questionnaires and active_questionnaires[disorder] == session_id:
            del active_questionnaires[disorder]
        
        return jsonify({
            "status": "success",
            "message": "Session cancelled successfully"
        })
    
    except Exception as e:
        logger.error(f"Error cancelling session: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to cancel session"
        }), 500

def start_questionnaire_api(host='localhost', port=5100):
    """Start the questionnaire API server"""
    try:
        logger.info(f"Starting questionnaire API server on {host}:{port}")
        app.run(host=host, port=port, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Error starting questionnaire API: {str(e)}")

def start_in_thread():
    """Start the API server in a separate thread"""
    thread = threading.Thread(target=start_questionnaire_api)
    thread.daemon = True
    thread.start()
    return thread

if __name__ == "__main__":
    # Start the API server directly
    start_questionnaire_api() 