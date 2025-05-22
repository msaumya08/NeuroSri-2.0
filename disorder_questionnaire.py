import json
import os
from datetime import datetime

class DisorderQuestionnaire:
    """
    Class to handle disorder-specific questionnaires when potential disorders are detected
    """
    
    def __init__(self, output_dir="questionnaire_results"):
        self.output_dir = output_dir
        self.questionnaires = self._load_questionnaires()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def _load_questionnaires(self):
        """
        Load questionnaire templates for different disorders
        """
        questionnaires = {
            "Generalized Anxiety Disorder": {
                "name": "GAD-7 (Generalized Anxiety Disorder 7-item scale)",
                "introduction": "This questionnaire helps assess anxiety symptoms. Please indicate how often you've been bothered by the following over the past 2 weeks:",
                "questions": [
                    {"id": 1, "text": "Feeling nervous, anxious, or on edge", "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]},
                    {"id": 2, "text": "Not being able to stop or control worrying", "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]},
                    {"id": 3, "text": "Worrying too much about different things", "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]},
                    {"id": 4, "text": "Trouble relaxing", "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]},
                    {"id": 5, "text": "Being so restless that it's hard to sit still", "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]},
                    {"id": 6, "text": "Becoming easily annoyed or irritable", "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]},
                    {"id": 7, "text": "Feeling afraid as if something awful might happen", "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]}
                ],
                "scoring": {
                    "Not at all": 0,
                    "Several days": 1,
                    "More than half the days": 2,
                    "Nearly every day": 3
                },
                "interpretation": {
                    "0-4": "Minimal anxiety",
                    "5-9": "Mild anxiety",
                    "10-14": "Moderate anxiety",
                    "15-21": "Severe anxiety"
                }
            },
            
            "Panic Disorder": {
                "name": "Panic Disorder Screener",
                "introduction": "This questionnaire helps identify potential panic disorder symptoms. Please answer yes or no to the following questions:",
                "questions": [
                    {"id": 1, "text": "In the past month, have you experienced sudden episodes of intense fear or discomfort?", "options": ["Yes", "No"]},
                    {"id": 2, "text": "Did these episodes reach their peak within minutes?", "options": ["Yes", "No"]},
                    {"id": 3, "text": "During these episodes, did you experience heart pounding or racing?", "options": ["Yes", "No"]},
                    {"id": 4, "text": "Did you feel short of breath or like you couldn't breathe properly?", "options": ["Yes", "No"]},
                    {"id": 5, "text": "Did you fear you were losing control or going crazy?", "options": ["Yes", "No"]},
                    {"id": 6, "text": "Did you experience fear of dying during these episodes?", "options": ["Yes", "No"]},
                    {"id": 7, "text": "Have you been worried about having another episode or changed your behavior because of them?", "options": ["Yes", "No"]}
                ],
                "scoring": {
                    "Yes": 1,
                    "No": 0
                },
                "interpretation": {
                    "0-1": "Unlikely to have panic disorder",
                    "2-3": "Possible panic symptoms",
                    "4-5": "Moderate likelihood of panic disorder",
                    "6-7": "High likelihood of panic disorder"
                }
            },
            
            "Major Depressive Disorder": {
                "name": "PHQ-9 (Patient Health Questionnaire 9-item scale)",
                "introduction": "This questionnaire helps assess depression symptoms. Over the last 2 weeks, how often have you been bothered by the following problems?",
                "questions": [
                    {"id": 1, "text": "Little interest or pleasure in doing things", "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]},
                    {"id": 2, "text": "Feeling down, depressed, or hopeless", "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]},
                    {"id": 3, "text": "Trouble falling or staying asleep, or sleeping too much", "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]},
                    {"id": 4, "text": "Feeling tired or having little energy", "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]},
                    {"id": 5, "text": "Poor appetite or overeating", "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]},
                    {"id": 6, "text": "Feeling bad about yourself or that you are a failure", "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]},
                    {"id": 7, "text": "Trouble concentrating on things", "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]},
                    {"id": 8, "text": "Moving or speaking so slowly that other people could have noticed, or being fidgety/restless", "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]},
                    {"id": 9, "text": "Thoughts that you would be better off dead or of hurting yourself", "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]}
                ],
                "scoring": {
                    "Not at all": 0,
                    "Several days": 1,
                    "More than half the days": 2,
                    "Nearly every day": 3
                },
                "interpretation": {
                    "0-4": "Minimal depression",
                    "5-9": "Mild depression",
                    "10-14": "Moderate depression",
                    "15-19": "Moderately severe depression",
                    "20-27": "Severe depression"
                }
            },
            
            "OCD": {
                "name": "OCD Screener",
                "introduction": "This questionnaire helps identify potential obsessive-compulsive symptoms. Please answer the following questions:",
                "questions": [
                    {"id": 1, "text": "Do you experience unwanted thoughts, images, or impulses that repeatedly enter your mind?", "options": ["Not at all", "A little", "Moderately", "A lot"]},
                    {"id": 2, "text": "Do you feel the need to perform certain behaviors or mental acts over and over?", "options": ["Not at all", "A little", "Moderately", "A lot"]},
                    {"id": 3, "text": "Do these thoughts or behaviors cause you significant distress?", "options": ["Not at all", "A little", "Moderately", "A lot"]},
                    {"id": 4, "text": "Do you spend more than 1 hour per day dealing with these thoughts or behaviors?", "options": ["Not at all", "A little", "Moderately", "A lot"]},
                    {"id": 5, "text": "Do these thoughts or behaviors interfere with your normal activities?", "options": ["Not at all", "A little", "Moderately", "A lot"]}
                ],
                "scoring": {
                    "Not at all": 0,
                    "A little": 1,
                    "Moderately": 2,
                    "A lot": 3
                },
                "interpretation": {
                    "0-3": "Unlikely to have OCD",
                    "4-7": "Mild OCD symptoms possible",
                    "8-11": "Moderate OCD symptoms likely",
                    "12-15": "Severe OCD symptoms"
                }
            },
            
            "ADHD": {
                "name": "Adult ADHD Self-Report Scale (ASRS)",
                "introduction": "This questionnaire helps identify potential ADHD symptoms in adults. Please answer how often each of the following happens to you:",
                "questions": [
                    {"id": 1, "text": "How often do you have trouble wrapping up final details of a project after the challenging parts have been done?", "options": ["Never", "Rarely", "Sometimes", "Often", "Very Often"]},
                    {"id": 2, "text": "How often do you have difficulty getting things in order when you have to do a task that requires organization?", "options": ["Never", "Rarely", "Sometimes", "Often", "Very Often"]},
                    {"id": 3, "text": "How often do you have problems remembering appointments or obligations?", "options": ["Never", "Rarely", "Sometimes", "Often", "Very Often"]},
                    {"id": 4, "text": "How often do you fidget or squirm with your hands or feet when you have to sit down for a long time?", "options": ["Never", "Rarely", "Sometimes", "Often", "Very Often"]},
                    {"id": 5, "text": "How often do you feel overly active and compelled to do things, like you were driven by a motor?", "options": ["Never", "Rarely", "Sometimes", "Often", "Very Often"]},
                    {"id": 6, "text": "How often do you find yourself talking too much when you are in social situations?", "options": ["Never", "Rarely", "Sometimes", "Often", "Very Often"]}
                ],
                "scoring": {
                    "Never": 0,
                    "Rarely": 1,
                    "Sometimes": 2,
                    "Often": 3,
                    "Very Often": 4
                },
                "interpretation": {
                    "0-9": "Unlikely to have ADHD",
                    "10-14": "Possible ADHD symptoms",
                    "15-19": "Likely ADHD symptoms",
                    "20-24": "Very likely ADHD symptoms"
                }
            }
        }
        
        return questionnaires
    
    def get_questionnaire(self, disorder):
        """
        Get the questionnaire for a specific disorder
        
        Args:
            disorder (str): The name of the disorder
            
        Returns:
            dict: The questionnaire data or None if not found
        """
        return self.questionnaires.get(disorder)
    
    def score_responses(self, disorder, responses):
        """
        Score a completed questionnaire
        
        Args:
            disorder (str): The name of the disorder
            responses (list): List of responses (indexes or text of selected options)
            
        Returns:
            dict: Scoring results including score, interpretation, and recommendations
        """
        questionnaire = self.get_questionnaire(disorder)
        if not questionnaire:
            return {"error": f"No questionnaire found for {disorder}"}
        
        # Initialize score
        total_score = 0
        
        # Score each response
        for i, response in enumerate(responses):
            if i >= len(questionnaire["questions"]):
                break
                
            # If response is an index, convert to option text
            if isinstance(response, int):
                response = questionnaire["questions"][i]["options"][response]
                
            # Add score for this response
            total_score += questionnaire["scoring"].get(response, 0)
        
        # Get interpretation
        interpretation = None
        for score_range, interp in questionnaire["interpretation"].items():
            # Parse score range (e.g., "5-9")
            if "-" in score_range:
                min_score, max_score = map(int, score_range.split("-"))
                if min_score <= total_score <= max_score:
                    interpretation = interp
                    break
        
        # Generate recommendations based on score
        recommendations = []
        if interpretation and "severe" in interpretation.lower() or "high" in interpretation.lower():
            recommendations.append("Consider consulting with a mental health professional")
            recommendations.append("This questionnaire suggests significant symptoms that may benefit from professional support")
        elif interpretation and ("moderate" in interpretation.lower() or "likely" in interpretation.lower()):
            recommendations.append("Consider speaking with a healthcare provider about these symptoms")
            recommendations.append("Some strategies may help manage these symptoms, but professional guidance is recommended")
        else:
            recommendations.append("Monitor your symptoms and consider discussing them with a healthcare provider if they persist or worsen")
        
        # Save results
        result = {
            "disorder": disorder,
            "questionnaire_name": questionnaire["name"],
            "score": total_score,
            "interpretation": interpretation,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file
        self._save_result(disorder, result)
        
        return result
    
    def _save_result(self, disorder, result):
        """Save questionnaire result to a file"""
        try:
            # Create a filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/{disorder.replace(' ', '_')}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving questionnaire result: {e}")
            return False
    
    def get_all_disorders(self):
        """Get a list of all available disorder questionnaires"""
        return list(self.questionnaires.keys())

# Example usage
if __name__ == "__main__":
    # Create questionnaire handler
    questionnaire = DisorderQuestionnaire()
    
    # Print available disorders
    print("Available questionnaires for disorders:")
    for disorder in questionnaire.get_all_disorders():
        print(f"- {disorder}")
    
    # Example: Get GAD questionnaire
    gad = questionnaire.get_questionnaire("Generalized Anxiety Disorder")
    if gad:
        print(f"\n{gad['name']}")
        print(gad['introduction'])
        for q in gad['questions']:
            print(f"\n{q['id']}. {q['text']}")
            for i, option in enumerate(q['options']):
                print(f"   {i}: {option}")
    
    # Example: Score a sample response
    sample_responses = [3, 2, 2, 1, 0, 1, 2]  # Example responses using option indexes
    result = questionnaire.score_responses("Generalized Anxiety Disorder", sample_responses)
    
    print("\nQuestionnaire Results:")
    print(f"Score: {result['score']}")
    print(f"Interpretation: {result['interpretation']}")
    print("Recommendations:")
    for rec in result['recommendations']:
        print(f"- {rec}") 