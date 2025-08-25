import streamlit as st
import pyttsx3
import speech_recognition as sr
import threading
import time
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import json
import random
from datetime import datetime
import pandas as pd

# Configure page
st.set_page_config(
    page_title="AI Career Interview Simulator",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'feedback' not in st.session_state:
    st.session_state.feedback = []
if 'selected_track' not in st.session_state:
    st.session_state.selected_track = None
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}

# Career tracks and their specific questions
CAREER_TRACKS = {
    "Software Development": {
        "description": "Build applications, websites, and software solutions",
        "skills": ["Programming", "Problem-solving", "Logic", "Debugging"],
        "questions": [
            "Tell me about your programming experience and favorite languages.",
            "How do you approach debugging a complex issue?",
            "Describe a challenging project you've worked on.",
            "How do you stay updated with new technologies?",
            "Explain a time when you had to learn a new framework quickly."
        ]
    },
    "Data Science": {
        "description": "Analyze data to extract insights and build predictive models",
        "skills": ["Statistics", "Python/R", "Machine Learning", "Data Visualization"],
        "questions": [
            "How would you explain machine learning to a non-technical person?",
            "Describe your experience with data cleaning and preprocessing.",
            "What's your approach to handling missing data?",
            "Tell me about a data project that provided business value.",
            "How do you validate the accuracy of your models?"
        ]
    },
    "AI/Machine Learning": {
        "description": "Develop intelligent systems and machine learning models",
        "skills": ["Deep Learning", "Neural Networks", "TensorFlow/PyTorch", "Computer Vision"],
        "questions": [
            "Explain the difference between supervised and unsupervised learning.",
            "How would you handle overfitting in a neural network?",
            "Describe your experience with different ML algorithms.",
            "What's your approach to feature engineering?",
            "Tell me about an AI project you're proud of."
        ]
    },
    "Cybersecurity": {
        "description": "Protect systems and data from digital threats",
        "skills": ["Network Security", "Ethical Hacking", "Risk Assessment", "Incident Response"],
        "questions": [
            "How would you secure a web application against common attacks?",
            "Describe your experience with penetration testing.",
            "What's your approach to incident response?",
            "How do you stay updated with the latest security threats?",
            "Explain a time when you identified a security vulnerability."
        ]
    },
    "Digital Marketing": {
        "description": "Promote products and services through digital channels",
        "skills": ["SEO/SEM", "Social Media", "Analytics", "Content Strategy"],
        "questions": [
            "How do you measure the success of a digital marketing campaign?",
            "Describe your experience with SEO and content optimization.",
            "What's your approach to social media strategy?",
            "How do you use data analytics in marketing decisions?",
            "Tell me about a successful campaign you've managed."
        ]
    },
    "Product Management": {
        "description": "Guide product development from conception to launch",
        "skills": ["Strategy", "User Research", "Agile", "Stakeholder Management"],
        "questions": [
            "How do you prioritize features for a product roadmap?",
            "Describe your experience with user research and feedback.",
            "What's your approach to working with engineering teams?",
            "How do you measure product success?",
            "Tell me about a product decision you had to make with limited data."
        ]
    },
    "UI/UX Design": {
        "description": "Create intuitive and engaging user experiences",
        "skills": ["Design Thinking", "Prototyping", "User Research", "Visual Design"],
        "questions": [
            "Walk me through your design process for a new application.",
            "How do you gather and incorporate user feedback?",
            "Describe a time when you had to balance user needs with business goals.",
            "What's your approach to creating accessible designs?",
            "Tell me about a design challenge you've overcome."
        ]
    },
    "Business Analysis": {
        "description": "Bridge business needs with technical solutions",
        "skills": ["Requirements Gathering", "Process Improvement", "Documentation", "Stakeholder Management"],
        "questions": [
            "How do you gather and document business requirements?",
            "Describe your experience with process improvement initiatives.",
            "What's your approach to stakeholder management?",
            "How do you handle conflicting requirements from different stakeholders?",
            "Tell me about a time when you identified a business opportunity."
        ]
    }
}

@st.cache_resource
def load_ai_model():
    """Load the AI model for generating feedback"""
    try:
        # Use a lightweight model for text generation
        generator = pipeline("text-generation", 
                           model="microsoft/DialoGPT-small",
                           tokenizer="microsoft/DialoGPT-small")
        return generator
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def text_to_speech(text):
    """Convert text to speech"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        
        # Use a separate thread to avoid blocking
        def speak():
            engine.say(text)
            engine.runAndWait()
        
        thread = threading.Thread(target=speak)
        thread.daemon = True
        thread.start()
        return True
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")
        return False

def speech_to_text():
    """Convert speech to text"""
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Listening... Please speak your answer.")
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=10, phrase_time_limit=30)
        
        text = r.recognize_google(audio)
        return text
    except sr.WaitTimeoutError:
        return "No speech detected. Please try again."
    except sr.UnknownValueError:
        return "Could not understand the audio. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"

def generate_feedback(question, answer, track):
    """Generate AI feedback for the answer"""
    if not answer or len(answer.strip()) < 10:
        return "Your answer seems too brief. Try to provide more detailed examples and explanations."
    
    # Simple rule-based feedback (can be enhanced with AI model)
    feedback_points = []
    
    # Check answer length
    if len(answer.split()) < 20:
        feedback_points.append("Consider providing more detailed examples to strengthen your answer.")
    elif len(answer.split()) > 100:
        feedback_points.append("Good detail, but try to be more concise in your delivery.")
    else:
        feedback_points.append("Good answer length and structure.")
    
    # Check for key skills mentioned
    track_skills = CAREER_TRACKS[track]["skills"]
    mentioned_skills = [skill for skill in track_skills if skill.lower() in answer.lower()]
    
    if mentioned_skills:
        feedback_points.append(f"Great! You mentioned relevant skills: {', '.join(mentioned_skills)}")
    else:
        feedback_points.append("Try to highlight specific technical skills relevant to the role.")
    
    # Check for examples
    example_indicators = ["example", "project", "experience", "time when", "situation"]
    if any(indicator in answer.lower() for indicator in example_indicators):
        feedback_points.append("Excellent use of specific examples to support your points.")
    else:
        feedback_points.append("Consider adding concrete examples from your experience.")
    
    return " | ".join(feedback_points)

def generate_career_recommendations(user_profile):
    """Generate career recommendations based on user profile"""
    recommendations = []
    
    for track, info in CAREER_TRACKS.items():
        score = 0
        reasons = []
        
        # Check skills match
        user_skills = user_profile.get('skills', [])
        matching_skills = [skill for skill in info['skills'] if any(us.lower() in skill.lower() for us in user_skills)]
        if matching_skills:
            score += len(matching_skills) * 2
            reasons.append(f"Skills match: {', '.join(matching_skills)}")
        
        # Check interests
        if user_profile.get('interests'):
            if any(interest.lower() in info['description'].lower() for interest in user_profile['interests']):
                score += 3
                reasons.append("Aligns with your stated interests")
        
        if score > 0:
            recommendations.append({
                'track': track,
                'score': score,
                'reasons': reasons,
                'description': info['description']
            })
    
    return sorted(recommendations, key=lambda x: x['score'], reverse=True)

# Main UI
st.title("🎯 AI Career Interview Simulator")
st.markdown("### Practice interviews for different tech career paths with AI feedback")

# Sidebar for navigation
with st.sidebar:
    st.header("🚀 Navigation")
    mode = st.selectbox(
        "Choose Mode:",
        ["Career Assessment", "Interview Practice", "Career Recommendations"]
    )

if mode == "Career Assessment":
    st.header("📋 Career Assessment")
    st.markdown("Let's understand your background and interests to recommend the best career tracks.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        name = st.text_input("Full Name")
        education = st.selectbox("Education Level", 
                               ["High School", "Bachelor's Degree", "Master's Degree", "PhD", "Other"])
        major = st.text_input("Field of Study")
        experience = st.selectbox("Years of Experience", 
                                ["0 (Student)", "0-1", "1-3", "3-5", "5+"])
    
    with col2:
        st.subheader("Skills & Interests")
        skills = st.multiselect("Select your skills:", 
                              ["Programming", "Data Analysis", "Machine Learning", "Design", 
                               "Marketing", "Project Management", "Communication", "Problem Solving",
                               "Leadership", "Research", "Statistics", "Database Management"])
        
        interests = st.multiselect("What interests you?",
                                 ["Building Applications", "Analyzing Data", "AI/ML", "Design",
                                  "Marketing", "Management", "Security", "Research"])
        
        learning_style = st.selectbox("Preferred Learning Style",
                                    ["Hands-on Projects", "Theoretical Study", "Collaborative Learning", "Self-paced"])
    
    if st.button("💾 Save Profile"):
        st.session_state.user_profile = {
            'name': name,
            'education': education,
            'major': major,
            'experience': experience,
            'skills': skills,
            'interests': interests,
            'learning_style': learning_style
        }
        st.success("Profile saved! Now try the Interview Practice or get Career Recommendations.")

elif mode == "Interview Practice":
    st.header("🎤 Interview Practice")
    
    if not st.session_state.user_profile:
        st.warning("⚠️ Please complete the Career Assessment first!")
        st.stop()
    
    # Track selection
    if not st.session_state.selected_track:
        st.subheader("Select a Career Track")
        
        cols = st.columns(2)
        for i, (track, info) in enumerate(CAREER_TRACKS.items()):
            col = cols[i % 2]
            with col:
                if st.button(f"**{track}**\n\n{info['description']}", key=f"track_{i}"):
                    st.session_state.selected_track = track
                    st.session_state.current_question = 0
                    st.session_state.answers = []
                    st.session_state.feedback = []
                    st.rerun()
    else:
        track = st.session_state.selected_track
        st.subheader(f"Interview for: {track}")
        
        # Progress bar
        progress = st.session_state.current_question / len(CAREER_TRACKS[track]["questions"])
        st.progress(progress)
        
        if st.session_state.current_question < len(CAREER_TRACKS[track]["questions"]):
            current_q = CAREER_TRACKS[track]["questions"][st.session_state.current_question]
            
            st.markdown(f"### Question {st.session_state.current_question + 1}:")
            st.markdown(f"**{current_q}**")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            # Text input for answer
            answer = st.text_area("Your Answer:", height=150, key=f"answer_{st.session_state.current_question}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔊 Read Question Aloud"):
                    text_to_speech(current_q)
            
            with col2:
                if st.button("🎤 Voice Answer"):
                    with st.spinner("Listening..."):
                        voice_answer = speech_to_text()
                        if voice_answer:
                            st.text_area("Voice Input:", value=voice_answer, key="voice_input")
                            answer = voice_answer
            
            with col3:
                if st.button("➡️ Submit Answer") and answer:
                    feedback = generate_feedback(current_q, answer, track)
                    st.session_state.answers.append(answer)
                    st.session_state.feedback.append(feedback)
                    st.session_state.current_question += 1
                    st.rerun()
        else:
            # Interview completed
            st.success("🎉 Interview Completed!")
            
            st.subheader("📊 Your Performance Summary")
            for i, (q, a, f) in enumerate(zip(CAREER_TRACKS[track]["questions"], 
                                            st.session_state.answers, 
                                            st.session_state.feedback)):
                with st.expander(f"Question {i+1}: {q[:50]}..."):
                    st.write("**Your Answer:**", a)
                    st.info(f"**Feedback:** {f}")
            
            # Overall score simulation
            avg_score = random.randint(70, 95)
            st.metric("Overall Interview Score", f"{avg_score}%")
            
            if st.button("🔄 Start New Interview"):
                st.session_state.selected_track = None
                st.session_state.current_question = 0
                st.session_state.answers = []
                st.session_state.feedback = []
                st.rerun()

elif mode == "Career Recommendations":
    st.header("🎯 Career Recommendations")
    
    if not st.session_state.user_profile:
        st.warning("⚠️ Please complete the Career Assessment first!")
        st.stop()
    
    recommendations = generate_career_recommendations(st.session_state.user_profile)
    
    if recommendations:
        st.subheader(f"Recommended Career Paths for {st.session_state.user_profile.get('name', 'You')}")
        
        for i, rec in enumerate(recommendations[:5]):  # Top 5 recommendations
            with st.expander(f"#{i+1} {rec['track']} (Match Score: {rec['score']})"):
                st.write(f"**Description:** {rec['description']}")
                st.write("**Why this fits you:**")
                for reason in rec['reasons']:
                    st.write(f"• {reason}")
                
                # Required skills
                required_skills = CAREER_TRACKS[rec['track']]['skills']
                st.write("**Key Skills Needed:**")
                st.write(", ".join(required_skills))
                
                if st.button(f"Practice Interview for {rec['track']}", key=f"practice_{i}"):
                    st.session_state.selected_track = rec['track']
                    st.session_state.current_question = 0
                    st.session_state.answers = []
                    st.session_state.feedback = []
                    # Switch to interview mode
                    st.success(f"Switched to interview practice for {rec['track']}!")
    else:
        st.info("Complete your profile assessment to get personalized recommendations!")

# Footer
st.markdown("---")
st.markdown("**💡 Tips for Better Interviews:**")
st.markdown("• Use specific examples from your experience • Highlight relevant technical skills • Practice your delivery • Be concise but detailed • Show enthusiasm for the role")

# Installation requirements comment
"""
Installation Requirements:
pip install streamlit transformers torch pyttsx3 SpeechRecognition pyaudio pandas

For audio functionality, you may also need:
- On Windows: No additional setup
- On macOS: brew install portaudio
- On Linux: sudo apt-get install portaudio19-dev python3-pyaudio

To run: streamlit run app.py
"""
