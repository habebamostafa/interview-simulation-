import streamlit as st
import requests
import json
import time
import random
from datetime import datetime
import pandas as pd

# Configure page
st.set_page_config(
    page_title="AI Interview Simulator",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'current_role' not in st.session_state:
    st.session_state.current_role = None
if 'question_count' not in st.session_state:
    st.session_state.question_count = 0
if 'max_questions' not in st.session_state:
    st.session_state.max_questions = 5
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}
if 'feedback_scores' not in st.session_state:
    st.session_state.feedback_scores = []

# Career roles configuration
CAREER_ROLES = {
    "Software Developer": {
        "icon": "💻",
        "description": "Build applications, websites, and software solutions",
        "focus_areas": ["programming languages", "frameworks", "problem-solving", "debugging", "system design"],
        "level_options": ["Junior", "Mid-level", "Senior", "Lead"]
    },
    "Data Scientist": {
        "icon": "📊", 
        "description": "Analyze data and build machine learning models",
        "focus_areas": ["statistics", "machine learning", "data analysis", "Python/R", "business insights"],
        "level_options": ["Junior", "Mid-level", "Senior", "Principal"]
    },
    "Product Manager": {
        "icon": "🚀",
        "description": "Guide product development and strategy",
        "focus_areas": ["product strategy", "user research", "roadmapping", "stakeholder management", "metrics"],
        "level_options": ["Associate", "Product Manager", "Senior PM", "Director"]
    },
    "UX Designer": {
        "icon": "🎨",
        "description": "Design user experiences and interfaces",
        "focus_areas": ["user research", "design thinking", "prototyping", "usability testing", "visual design"],
        "level_options": ["Junior", "Mid-level", "Senior", "Lead"]
    },
    "DevOps Engineer": {
        "icon": "⚙️",
        "description": "Manage infrastructure and deployment pipelines",
        "focus_areas": ["cloud platforms", "CI/CD", "containerization", "monitoring", "automation"],
        "level_options": ["Junior", "Mid-level", "Senior", "Principal"]
    },
    "Cybersecurity Analyst": {
        "icon": "🛡️",
        "description": "Protect systems from security threats",
        "focus_areas": ["threat analysis", "incident response", "security tools", "compliance", "risk assessment"],
        "level_options": ["Analyst", "Senior Analyst", "Specialist", "Manager"]
    },
    "Marketing Manager": {
        "icon": "📱",
        "description": "Develop and execute marketing strategies",
        "focus_areas": ["digital marketing", "campaign management", "analytics", "brand strategy", "customer acquisition"],
        "level_options": ["Coordinator", "Manager", "Senior Manager", "Director"]
    },
    "Business Analyst": {
        "icon": "📈",
        "description": "Analyze business processes and requirements",
        "focus_areas": ["requirements gathering", "process improvement", "stakeholder management", "documentation", "analysis"],
        "level_options": ["Junior", "Business Analyst", "Senior BA", "Lead"]
    }
}

def call_huggingface_api(prompt, hf_token, max_length=500):
    """Call Hugging Face API with the given prompt"""
    
    API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": max_length,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '').strip()
            return result.get('generated_text', '').strip()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        return None

def generate_interview_question(role, level, question_number, focus_areas, hf_token):
    """Generate an interview question using AI"""
    
    prompt = f"""You are an experienced {role} hiring manager conducting a professional interview. 

Role: {role} ({level} level)
Question number: {question_number} out of 5
Focus areas: {', '.join(focus_areas)}

Generate a realistic, professional interview question that:
1. Is appropriate for {level} level candidates
2. Tests practical knowledge and experience
3. Is commonly asked in {role} interviews
4. Focuses on one of these areas: {', '.join(focus_areas)}
5. Encourages the candidate to share specific examples

Format your response as just the interview question, without any additional text or context.

Question:"""

    return call_huggingface_api(prompt, hf_token, max_length=150)

def generate_feedback(question, answer, role, level, hf_token):
    """Generate feedback for the candidate's answer"""
    
    prompt = f"""You are an expert {role} interviewer evaluating a candidate's response.

Role: {role} ({level} level)
Question: {question}
Candidate's Answer: {answer}

Provide professional feedback that includes:
1. What the candidate did well
2. Areas for improvement
3. Missing elements that should be included
4. A score out of 10
5. Specific suggestions for better responses

Keep the feedback constructive, professional, and actionable. Focus on the technical and soft skills demonstrated.

Feedback:"""

    return call_huggingface_api(prompt, hf_token, max_length=400)

def generate_follow_up_question(original_question, answer, role, hf_token):
    """Generate a follow-up question based on the candidate's answer"""
    
    prompt = f"""You are interviewing for a {role} position. Based on the candidate's previous answer, generate a relevant follow-up question.

Original Question: {original_question}
Candidate's Answer: {answer}

Generate a natural follow-up question that:
1. Digs deeper into their response
2. Tests their knowledge further
3. Is conversational and professional
4. Explores practical application

Follow-up Question:"""

    return call_huggingface_api(prompt, hf_token, max_length=100)

def extract_score_from_feedback(feedback_text):
    """Extract numerical score from feedback text"""
    import re
    
    # Look for patterns like "score: 8/10", "8 out of 10", "score of 8"
    patterns = [
        r'score[:\s]*(\d+)[/\s]*10',
        r'(\d+)[/\s]*10',
        r'score of (\d+)',
        r'rating[:\s]*(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, feedback_text.lower())
        if match:
            return int(match.group(1))
    
    # Default scoring based on keywords
    if 'excellent' in feedback_text.lower() or 'outstanding' in feedback_text.lower():
        return 9
    elif 'good' in feedback_text.lower() or 'well' in feedback_text.lower():
        return 7
    elif 'adequate' in feedback_text.lower() or 'satisfactory' in feedback_text.lower():
        return 6
    elif 'needs improvement' in feedback_text.lower():
        return 4
    else:
        return 5

# CSS Styling
st.markdown("""
<style>
.interview-container {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
.question-box {
    background-color: #e3f2fd;
    padding: 15px;
    border-left: 4px solid #2196f3;
    border-radius: 5px;
    margin: 10px 0;
}
.feedback-box {
    background-color: #f3e5f5;
    padding: 15px;
    border-left: 4px solid #9c27b0;
    border-radius: 5px;
    margin: 10px 0;
}
.score-excellent { color: #4caf50; font-weight: bold; }
.score-good { color: #ff9800; font-weight: bold; }
.score-needs-improvement { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Main UI
st.title("🎯 AI-Powered Interview Simulator")
st.markdown("### Practice real interviews with AI feedback - Powered by Hugging Face")

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Token input
    hf_token = st.secrets.get("hf_token", None)
    
    if not hf_token:
        st.warning("⚠️ Please enter your Hugging Face token to continue")
        st.markdown("[Get your free token here](https://huggingface.co/settings/tokens)")
        st.stop()
    
    st.markdown("---")
    
    # Interview Settings
    st.subheader("⚙️ Interview Settings")
    max_questions = st.slider("Number of Questions", 3, 8, 5)
    st.session_state.max_questions = max_questions
    
    # Progress tracking
    if st.session_state.interview_started:
        progress = st.session_state.question_count / st.session_state.max_questions
        st.progress(progress)
        st.write(f"Question {st.session_state.question_count}/{st.session_state.max_questions}")

# Main content
if not st.session_state.interview_started:
    st.markdown("## 🚀 Choose Your Interview Setup")
    
    # Role selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Select Role")
        selected_role = st.selectbox(
            "Choose the position you want to interview for:",
            list(CAREER_ROLES.keys()),
            format_func=lambda x: f"{CAREER_ROLES[x]['icon']} {x}"
        )
        
        role_info = CAREER_ROLES[selected_role]
        st.info(f"**Focus Areas:** {', '.join(role_info['focus_areas'])}")
        st.write(f"**Description:** {role_info['description']}")
    
    with col2:
        st.subheader("Experience Level")
        selected_level = st.selectbox(
            "Select your level:",
            role_info['level_options']
        )
        
        # Optional: Candidate info
        st.subheader("Optional Info")
        candidate_name = st.text_input("Your Name (optional)")
        years_exp = st.number_input("Years of Experience", 0, 20, 0)
    
    # Start interview button
    if st.button("🎬 Start Interview", type="primary", use_container_width=True):
        st.session_state.current_role = selected_role
        st.session_state.current_level = selected_level
        st.session_state.interview_started = True
        st.session_state.question_count = 1
        st.session_state.conversation_history = []
        st.session_state.feedback_scores = []
        st.session_state.user_profile = {
            'name': candidate_name,
            'role': selected_role,
            'level': selected_level,
            'years_exp': years_exp,
            'start_time': datetime.now()
        }
        st.rerun()

else:
    # Active interview
    role = st.session_state.current_role
    level = st.session_state.current_level
    role_info = CAREER_ROLES[role]
    
    # Interview header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"## {role_info['icon']} {role} Interview ({level} Level)")
    with col2:
        if st.button("🏠 End Interview"):
            st.session_state.interview_started = False
            st.rerun()
    
    # Generate and display question
    if st.session_state.question_count <= st.session_state.max_questions:
        
        # Generate question if not already generated
        current_question_key = f"question_{st.session_state.question_count}"
        if current_question_key not in st.session_state:
            with st.spinner("🤖 Generating interview question..."):
                question = generate_interview_question(
                    role, level, st.session_state.question_count, 
                    role_info['focus_areas'], hf_token
                )
                if question:
                    st.session_state[current_question_key] = question
                else:
                    st.error("Failed to generate question. Please check your API token and try again.")
                    st.stop()
        
        current_question = st.session_state[current_question_key]
        
        # Display question
        st.markdown('<div class="question-box">', unsafe_allow_html=True)
        st.markdown(f"### 💬 Question {st.session_state.question_count}")
        st.markdown(f"**{current_question}**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Text-to-speech simulation (since we can't use actual TTS on Streamlit Cloud)
        if st.button("🔊 Read Question Aloud"):
            st.info("🎵 Text-to-speech simulation: The question would be read aloud in a real implementation")
            st.audio("https://www.soundjay.com/misc/sounds/bell-ringing-05.wav", format="audio/wav")
        
        # Answer input
        st.markdown("### ✍️ Your Answer:")
        
        # Voice input simulation
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🎤 Voice Input"):
                st.info("🎵 Voice input simulation: In a real implementation, this would record your voice")
                st.session_state.voice_mode = True
        
        # Answer text area
        answer = st.text_area(
            "Type your response here:",
            height=200,
            key=f"answer_{st.session_state.question_count}",
            placeholder="Share your experience with specific examples..."
        )
        
        # Answer metrics
        if answer:
            word_count = len(answer.split())
            char_count = len(answer)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Words", word_count)
            with col2:
                st.metric("Characters", char_count)
            with col3:
                if word_count < 50:
                    st.markdown('<span style="color: orange;">⚠️ Consider more detail</span>', unsafe_allow_html=True)
                elif word_count > 300:
                    st.markdown('<span style="color: orange;">⚠️ Try to be more concise</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color: green;">✅ Good length</span>', unsafe_allow_html=True)
        
        # Submit answer
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("➡️ Submit Answer", type="primary", disabled=not answer):
                if len(answer.strip()) < 20:
                    st.error("Please provide a more detailed answer (at least 20 characters)")
                else:
                    # Store answer
                    st.session_state.conversation_history.append({
                        'question': current_question,
                        'answer': answer,
                        'question_number': st.session_state.question_count
                    })
                    
                    # Generate feedback
                    with st.spinner("🤖 Generating feedback..."):
                        feedback = generate_feedback(current_question, answer, role, level, hf_token)
                        if feedback:
                            # Extract score
                            score = extract_score_from_feedback(feedback)
                            st.session_state.feedback_scores.append(score)
                            
                            # Display feedback
                            st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
                            st.markdown("### 📝 AI Feedback:")
                            st.write(feedback)
                            
                            # Score visualization
                            if score >= 8:
                                st.markdown(f'<div class="score-excellent">Score: {score}/10 - Excellent!</div>', unsafe_allow_html=True)
                            elif score >= 6:
                                st.markdown(f'<div class="score-good">Score: {score}/10 - Good</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="score-needs-improvement">Score: {score}/10 - Needs Improvement</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Store feedback
                            st.session_state.conversation_history[-1]['feedback'] = feedback
                            st.session_state.conversation_history[-1]['score'] = score
                            
                            # Progress to next question
                            time.sleep(2)  # Brief pause to read feedback
                            st.session_state.question_count += 1
                            st.rerun()
                        else:
                            st.error("Failed to generate feedback. Please try again.")
        
        with col2:
            if st.session_state.question_count > 1:
                if st.button("⏭️ Skip Question"):
                    st.session_state.question_count += 1
                    st.rerun()
    
    else:
        # Interview completed
        st.success("🎉 Interview Completed!")
        
        # Calculate final score
        if st.session_state.feedback_scores:
            avg_score = sum(st.session_state.feedback_scores) / len(st.session_state.feedback_scores)
        else:
            avg_score = 0
        
        # Results summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Score", f"{avg_score:.1f}/10")
        with col2:
            grade = "A" if avg_score >= 9 else "B" if avg_score >= 7 else "C" if avg_score >= 5 else "D"
            st.metric("Grade", grade)
        with col3:
            duration = datetime.now() - st.session_state.user_profile['start_time']
            st.metric("Duration", f"{duration.seconds//60} min")
        
        # Detailed results
        st.markdown("## 📊 Interview Summary")
        
        for i, conv in enumerate(st.session_state.conversation_history):
            with st.expander(f"Question {i+1} - Score: {conv.get('score', 'N/A')}/10"):
                st.markdown(f"**Question:** {conv['question']}")
                st.markdown(f"**Your Answer:** {conv['answer']}")
                if 'feedback' in conv:
                    st.markdown(f"**Feedback:** {conv['feedback']}")
        
        # Download results
        if st.button("📥 Download Interview Report"):
            report_data = {
                'candidate': st.session_state.user_profile.get('name', 'Anonymous'),
                'role': role,
                'level': level,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'overall_score': avg_score,
                'questions_answered': len(st.session_state.conversation_history),
                'conversation': st.session_state.conversation_history
            }
            
            st.download_button(
                label="📋 Download as JSON",
                data=json.dumps(report_data, indent=2),
                file_name=f"interview_report_{role.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
        
        # Start new interview
        if st.button("🔄 Start New Interview", type="primary"):
            # Reset all session state
            for key in list(st.session_state.keys()):
                if key.startswith('question_') or key.startswith('answer_'):
                    del st.session_state[key]
            
            st.session_state.interview_started = False
            st.session_state.question_count = 0
            st.session_state.conversation_history = []
            st.session_state.feedback_scores = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("**💡 Interview Tips:**")
st.markdown("• Use the STAR method (Situation, Task, Action, Result) • Give specific examples • Be concise but thorough • Show your problem-solving process")

# Requirements as comment for easy setup
"""
Required packages for requirements.txt:

streamlit
requests
pandas

To get Hugging Face token:
1. Go to https://huggingface.co/
2. Create a free account
3. Go to Settings -> Access Tokens
4. Create a new token with 'read' permissions
5. Copy and paste the token in the app

To deploy on Streamlit Cloud:
1. Push this code to GitHub
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Deploy!

No additional audio libraries needed - the app simulates TTS/STT functionality
"""
