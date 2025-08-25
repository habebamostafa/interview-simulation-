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
if 'use_fallback_mode' not in st.session_state:
    st.session_state.use_fallback_mode = False
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'follow_up_mode' not in st.session_state:
    st.session_state.follow_up_mode = False
if 'follow_up_count' not in st.session_state:
    st.session_state.follow_up_count = 0

# Career roles configuration with predefined questions
CAREER_ROLES = {
    "Software Developer": {
        "icon": "💻",
        "description": "Build applications, websites, and software solutions",
        "focus_areas": ["programming languages", "frameworks", "problem-solving", "debugging", "system design"],
        "level_options": ["Student", "Junior", "Mid-level", "Senior", "Lead"],
        "questions": {
            "Student": [
                "What programming languages have you learned in school or on your own?",
                "Can you describe a programming project you've worked on outside of class?",
                "How do you approach learning new programming concepts?",
                "What development tools (IDEs, version control) are you familiar with?",
                "What area of software development interests you the most and why?"
            ],
            "Junior": [
                "What programming languages are you most comfortable with and why?",
                "Can you explain the concept of object-oriented programming?",
                "How would you approach debugging a piece of code that isn't working as expected?",
                "What is version control and why is it important in software development?",
                "Describe a simple project you've worked on and what you learned from it."
            ],
            "Mid-level": [
                "How do you ensure the quality of your code before submitting it for review?",
                "Can you explain the difference between REST and GraphQL APIs?",
                "Describe your experience with testing frameworks and methodologies.",
                "How would you optimize a slow database query?",
                "What's your approach to learning new technologies or frameworks?"
            ],
            "Senior": [
                "Describe your experience with system architecture and design patterns.",
                "How do you approach mentoring junior developers on your team?",
                "What strategies do you use for managing technical debt?",
                "Can you discuss a challenging technical problem you solved and your process?",
                "How do you balance business requirements with technical best practices?"
            ],
            "Lead": [
                "How do you approach technical leadership and decision-making in your team?",
                "Describe your experience with project planning and estimation.",
                "What strategies do you use for managing stakeholder expectations?",
                "How do you foster innovation and continuous improvement in your team?",
                "Can you discuss a time when you had to make a difficult technical trade-off?"
            ]
        }
    },
    "Data Scientist": {
        "icon": "📊", 
        "description": "Analyze data and build machine learning models",
        "focus_areas": ["statistics", "machine learning", "data analysis", "Python/R", "business insights"],
        "level_options": ["Student", "Junior", "Mid-level", "Senior", "Principal"],
        "questions": {
            "Student": [
                "What statistics or data science courses have you found most valuable?",
                "Can you describe a data analysis project you've worked on for class or personally?",
                "What data visualization tools or libraries are you familiar with?",
                "How do you approach cleaning and preparing data for analysis?",
                "What area of data science interests you the most and why?"
            ],
            "Junior": [
                "What statistical concepts are most important in data science?",
                "Can you explain the difference between supervised and unsupervised learning?",
                "What Python libraries are you familiar with for data analysis?",
                "How would you handle missing values in a dataset?",
                "Describe a simple data visualization you created and what it showed."
            ],
            "Mid-level": [
                "How do you evaluate the performance of a machine learning model?",
                "Can you explain the bias-variance tradeoff?",
                "What's your experience with feature engineering?",
                "How would you explain a complex model to non-technical stakeholders?",
                "Describe your process for cleaning and preparing data for analysis."
            ],
            "Senior": [
                "How do you approach designing a machine learning system from end to end?",
                "What strategies do you use for managing data science projects?",
                "Can you discuss a time when you had to make a trade-off between model complexity and interpretability?",
                "How do you stay updated with the latest developments in data science?",
                "What's your experience with deploying models to production environments?"
            ],
            "Principal": [
                "How do you develop data strategy for an organization?",
                "What's your approach to building and leading a data science team?",
                "How do you measure the ROI of data science initiatives?",
                "Can you discuss your experience with big data technologies?",
                "How do you ensure ethical use of data and AI in your projects?"
            ]
        }
    },
    "Product Manager": {
        "icon": "📝",
        "description": "Define product vision and work with teams to build successful products",
        "focus_areas": ["product strategy", "user research", "prioritization", "stakeholder management", "metrics"],
        "level_options": ["Student", "Junior", "Mid-level", "Senior", "Director"],
        "questions": {
            "Student": [
                "What makes a product successful in your opinion?",
                "How do you stay informed about technology trends and new products?",
                "Describe a product you admire and explain why it's well-designed.",
                "What techniques would you use to understand user needs?",
                "Have you ever managed a project from concept to completion?"
            ],
            "Junior": [
                "How would you prioritize between different feature requests?",
                "What metrics would you track to measure a product's success?",
                "How do you gather and incorporate user feedback?",
                "Describe your experience working with cross-functional teams.",
                "How do you create a product roadmap?"
            ],
            "Mid-level": [
                "How do you balance user needs with business objectives?",
                "Describe your approach to competitive analysis.",
                "How do you handle disagreements with engineering about feasibility?",
                "What's your experience with A/B testing and experimentation?",
                "How do you decide when to pivot a product strategy?"
            ],
            "Senior": [
                "How do you develop a product vision and strategy?",
                "Describe your experience managing a product through its lifecycle.",
                "How do you align stakeholders with different priorities?",
                "What's your approach to building and mentoring product teams?",
                "How do you measure and communicate product success to executives?"
            ],
            "Director": [
                "How do you develop product portfolio strategy?",
                "What's your approach to resource allocation across multiple products?",
                "How do you foster innovation across product teams?",
                "Describe your experience with M&A product integration.",
                "How do you build a product-driven culture in an organization?"
            ]
        }
    },
    "UX Designer": {
        "icon": "🎨",
        "description": "Create user-centered designs and improve user experiences",
        "focus_areas": ["user research", "wireframing", "prototyping", "usability testing", "design systems"],
        "level_options": ["Student", "Junior", "Mid-level", "Senior", "Lead"],
        "questions": {
            "Student": [
                "What design tools are you most comfortable with?",
                "Describe a design project you've worked on for class or personally.",
                "How do you approach understanding user needs?",
                "What makes a user interface intuitive in your opinion?",
                "Which designers or design philosophies inspire you?"
            ],
            "Junior": [
                "Walk me through your design process from concept to final design.",
                "How do you incorporate user feedback into your designs?",
                "What's your experience with creating wireframes and prototypes?",
                "How do you balance user needs with business requirements?",
                "Describe a time you had to iterate on a design based on feedback."
            ],
            "Mid-level": [
                "How do you approach designing for accessibility?",
                "Describe your experience with usability testing.",
                "How do you collaborate with developers during implementation?",
                "What's your experience with design systems?",
                "How do you measure the success of your designs?"
            ],
            "Senior": [
                "How do you develop a UX strategy for a product?",
                "Describe your experience mentoring other designers.",
                "How do you evangelize user-centered design in an organization?",
                "What's your approach to design critique and feedback?",
                "How do you stay updated with design trends and best practices?"
            ],
            "Lead": [
                "How do you build and manage a design team?",
                "What's your approach to establishing design processes?",
                "How do you align design strategy with business goals?",
                "Describe your experience with design operations.",
                "How do you measure the impact of design on business outcomes?"
            ]
        }
    }
}

def call_huggingface_api(prompt, model_name="google/flan-t5-large", max_length=500):
    """Call Hugging Face API with the given prompt"""
    
    # Get the token from secrets
    hf_token = st.secrets.get("hf_tokens", None)
    
    if not hf_token:
        st.error("Hugging Face token not found. Please add it to your Streamlit secrets.")
        return None
    
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
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
            # Switch to fallback mode if API is not available
            if response.status_code == 400 and "paused" in response.text:
                st.session_state.use_fallback_mode = True
                st.warning("Switching to fallback mode with predefined questions.")
            return None
    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        st.session_state.use_fallback_mode = True
        st.warning("Switching to fallback mode with predefined questions.")
        return None

def generate_interview_question(role, level, question_number, focus_areas, conversation_history):
    """Generate an interview question using AI or fallback to predefined questions"""
    
    # Use fallback mode if API is not available
    if st.session_state.use_fallback_mode:
        role_data = CAREER_ROLES.get(role, {})
        questions = role_data.get("questions", {}).get(level, [])
        
        if questions and question_number <= len(questions):
            return questions[question_number - 1]
        else:
            return "Tell me about your experience and why you're interested in this role."
    
    # Otherwise use AI generation
    history_context = ""
    if conversation_history:
        history_context = "Previous questions and answers:\n"
        for i, conv in enumerate(conversation_history[-3:]):  # Last 3 exchanges
            history_context += f"Q: {conv['question']}\nA: {conv['answer']}\n"
    
    prompt = f"""You are an experienced {role} hiring manager conducting a professional interview. 

Role: {role} ({level} level)
Question number: {question_number} out of 5
Focus areas: {', '.join(focus_areas)}
{history_context}

Generate a realistic, professional interview question that:
1. Is appropriate for {level} level candidates
2. Tests practical knowledge and experience
3. Is commonly asked in {role} interviews
4. Focuses on one of these areas: {', '.join(focus_areas)}
5. Encourages the candidate to share specific examples
6. Is different from previous questions

Format your response as just the interview question, without any additional text or context.

Question:"""

    return call_huggingface_api(prompt, max_length=150)

def generate_feedback(question, answer, role, level):
    """Generate feedback for the candidate's answer"""
    
    # Use fallback mode if API is not available
    if st.session_state.use_fallback_mode:
        return f"Thank you for your answer. For a {level} {role} position, we look for candidates who can demonstrate practical experience with specific examples. Consider expanding on your answer with more details about your direct experience and the impact of your work."
    
    # Otherwise use AI generation
    prompt = f"""You are an expert {role} interviewer evaluating a candidate's response.

Role: {role} ({level} level)
Question: {question}
Candidate's Answer: {answer}

Provide professional feedback that includes:
1. What the candidate did well (be specific)
2. Areas for improvement (be constructive)
3. Missing elements that should be included
4. A score out of 10 with justification
5. Specific suggestions for better responses

Keep the feedback constructive, professional, and actionable. Focus on the technical and soft skills demonstrated.

Feedback:"""

    return call_huggingface_api(prompt, max_length=400)

def generate_follow_up_question(original_question, answer, role, level):
    """Generate a follow-up question based on the candidate's answer"""
    
    # Use fallback mode if API is not available
    if st.session_state.use_fallback_mode:
        follow_ups = [
            "Can you tell me more about that experience?",
            "How did you handle the challenges you mentioned?",
            "What was the outcome of that project?",
            "What did you learn from that situation?",
            "How would you approach that differently today?"
        ]
        return random.choice(follow_ups)
    
    # Otherwise use AI generation
    prompt = f"""You are interviewing for a {role} position at {level} level. Based on the candidate's previous answer, generate a relevant follow-up question.

Original Question: {original_question}
Candidate's Answer: {answer}

Generate a natural follow-up question that:
1. Digs deeper into their specific response
2. Tests their knowledge further on the same topic
3. Is conversational and professional
4. Explores practical application or lessons learned
5. Is appropriate for a {level} level candidate

Follow-up Question:"""

    return call_huggingface_api(prompt, max_length=100)

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
.followup-box {
    background-color: #fff3e0;
    padding: 15px;
    border-left: 4px solid #ff9800;
    border-radius: 5px;
    margin: 10px 0;
}
.score-excellent { color: #4caf50; font-weight: bold; }
.score-good { color: #ff9800; font-weight: bold; }
.score-needs-improvement { color: #f44336; font-weight: bold; }
.role-card {
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    background-color: #f5f5f5;
    border-left: 5px solid #4caf50;
}
.student-highlight {
    background-color: #e8f5e9;
    padding: 10px;
    border-radius: 5px;
    border-left: 4px solid #4caf50;
}
</style>
""", unsafe_allow_html=True)

# Main UI
st.title("🎯 AI-Powered Interview Simulator")
st.markdown("### Practice real interviews with AI feedback - Powered by Hugging Face")

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Check if token exists
    hf_token = st.secrets.get("hf_tokens", None)
    
    if not hf_token:
        st.warning("⚠️ Please add your Hugging Face token to Streamlit secrets")
        st.markdown("[Get your free token here](https://huggingface.co/settings/tokens)")
        st.markdown("[How to add secrets to Streamlit](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)")
    else:
        st.success("✅ Hugging Face token found")
    
    st.markdown("---")
    
    # Interview Settings
    st.subheader("⚙️ Interview Settings")
    max_questions = st.slider("Number of Questions", 3, 8, 5)
    st.session_state.max_questions = max_questions
    
    # Enable follow-up questions
    enable_followup = st.checkbox("Enable follow-up questions", value=True)
    
    # Progress tracking
    if st.session_state.interview_started:
        progress = st.session_state.question_count / st.session_state.max_questions
        st.progress(progress)
        st.write(f"Question {st.session_state.question_count}/{st.session_state.max_questions}")

# Main content
if not st.session_state.interview_started:
    st.markdown("## 🚀 Choose Your Interview Setup")
    
    # Highlight student option
    st.markdown('<div class="student-highlight">🎓 <strong>Special student track available</strong> - Perfect for those preparing for internships or entry-level positions</div>', unsafe_allow_html=True)
    st.markdown("")
    
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
        
        # Display role information in a card
        st.markdown('<div class="role-card">', unsafe_allow_html=True)
        st.markdown(f"**{role_info['icon']} {selected_role}**")
        st.write(f"{role_info['description']}")
        st.markdown(f"**Focus Areas:** {', '.join(role_info['focus_areas'])}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Experience Level")
        selected_level = st.selectbox(
            "Select your level:",
            role_info['level_options']
        )
        
        # Show special message for students
        if selected_level == "Student":
            st.info("🎓 Great choice! This track is designed for students preparing for internships and entry-level positions.")
        
        # Optional: Candidate info
        st.subheader("Optional Info")
        candidate_name = st.text_input("Your Name (optional)")
        if selected_level != "Student":
            years_exp = st.number_input("Years of Experience", 0, 20, 2)
        else:
            years_exp = st.number_input("Years of Study", 0, 6, 2)
            st.caption("Years of academic study in this field")
    
    # Start interview button
    if st.button("🎬 Start Interview", type="primary", use_container_width=True):
        st.session_state.current_role = selected_role
        st.session_state.current_level = selected_level
        st.session_state.interview_started = True
        st.session_state.question_count = 1
        st.session_state.conversation_history = []
        st.session_state.feedback_scores = []
        st.session_state.follow_up_mode = False
        st.session_state.follow_up_count = 0
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
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f"## {role_info['icon']} {role} Interview ({level} Level)")
    with col2:
        if st.session_state.user_profile.get('name'):
            st.markdown(f"**Candidate:** {st.session_state.user_profile['name']}")
    with col3:
        if st.button("🏠 End Interview"):
            st.session_state.interview_started = False
            st.rerun()
    
    # Display mode info if in fallback mode
    if st.session_state.use_fallback_mode:
        st.warning("⚠️ Using fallback mode with predefined questions. API is currently unavailable.")
    
    # Generate and display question
    if st.session_state.question_count <= st.session_state.max_questions:
        
        # Check if we're in follow-up mode
        if st.session_state.follow_up_mode and st.session_state.follow_up_count < 2:
            # Generate follow-up question
            last_exchange = st.session_state.conversation_history[-1]
            with st.spinner("🤖 Generating follow-up question..."):
                follow_up_question = generate_follow_up_question(
                    last_exchange['question'], last_exchange['answer'], role, level
                )
                
                if follow_up_question:
                    st.session_state.current_question = follow_up_question
                    st.session_state.follow_up_count += 1
                else:
                    st.session_state.follow_up_mode = False
                    st.session_state.question_count += 1
                    st.rerun()
        else:
            # Generate new question
            st.session_state.follow_up_mode = False
            st.session_state.follow_up_count = 0
            
            with st.spinner("🤖 Generating interview question..."):
                question = generate_interview_question(
                    role, level, st.session_state.question_count, 
                    role_info['focus_areas'], st.session_state.conversation_history
                )
                if question:
                    st.session_state.current_question = question
                else:
                    # If API fails and we're not already in fallback mode
                    if not st.session_state.use_fallback_mode:
                        st.session_state.use_fallback_mode = True
                        st.rerun()
                    else:
                        st.error("Failed to generate question. Please try again.")
                        st.stop()
        
        current_question = st.session_state.current_question
        
        # Display question with appropriate styling
        if st.session_state.follow_up_mode:
            st.markdown('<div class="followup-box">', unsafe_allow_html=True)
            st.markdown(f"### 🔍 Follow-up Question")
        else:
            st.markdown('<div class="question-box">', unsafe_allow_html=True)
            st.markdown(f"### 💬 Question {st.session_state.question_count}")
        
        st.markdown(f"**{current_question}**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Text-to-speech simulation
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
            key=f"answer_{st.session_state.question_count}_{st.session_state.follow_up_count}",
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
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            submit_text = "➡️ Submit Answer" if not st.session_state.follow_up_mode else "➡️ Submit Follow-up"
            if st.button(submit_text, type="primary", disabled=not answer):
                if len(answer.strip()) < 20:
                    st.error("Please provide a more detailed answer (at least 20 characters)")
                else:
                    # Store answer
                    st.session_state.conversation_history.append({
                        'question': current_question,
                        'answer': answer,
                        'question_number': st.session_state.question_count,
                        'is_follow_up': st.session_state.follow_up_mode
                    })
                    
                    # Generate feedback
                    with st.spinner("🤖 Generating feedback..."):
                        feedback = generate_feedback(current_question, answer, role, level)
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
                            
                            # Decide whether to ask a follow-up question
                            if enable_followup and not st.session_state.follow_up_mode and score < 9 and st.session_state.follow_up_count < 2:
                                st.session_state.follow_up_mode = True
                                time.sleep(2)
                                st.rerun()
                            else:
                                # Progress to next question
                                time.sleep(2)  # Brief pause to read feedback
                                st.session_state.question_count += 1
                                st.session_state.follow_up_mode = False
                                st.session_state.follow_up_count = 0
                                st.rerun()
                        else:
                            st.error("Failed to generate feedback. Please try again.")
        
        with col2:
            if st.session_state.question_count > 1 and not st.session_state.follow_up_mode:
                if st.button("⏭️ Skip Question"):
                    st.session_state.question_count += 1
                    st.session_state.follow_up_mode = False
                    st.session_state.follow_up_count = 0
                    st.rerun()
        
        with col3:
            if st.session_state.follow_up_mode:
                if st.button("↩️ Skip Follow-up"):
                    st.session_state.question_count += 1
                    st.session_state.follow_up_mode = False
                    st.session_state.follow_up_count = 0
                    st.rerun()
    
    else:
        # Interview completed
        st.success("🎉 Interview Completed!")
        st.balloons()
        
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
            if avg_score >= 8.5:
                grade = "A"
            elif avg_score >= 7:
                grade = "B"
            elif avg_score >= 5.5:
                grade = "C"
            else:
                grade = "Needs Practice"
            st.metric("Grade", grade)
        with col3:
            duration = datetime.now() - st.session_state.user_profile['start_time']
            st.metric("Duration", f"{duration.seconds//60} min")
        
        # Performance analysis
        st.markdown("## 📊 Performance Analysis")
        
        # Create a simple chart of scores
        if st.session_state.feedback_scores:
            scores_df = pd.DataFrame({
                'Question': range(1, len(st.session_state.feedback_scores) + 1),
                'Score': st.session_state.feedback_scores
            })
            st.line_chart(scores_df.set_index('Question'))
        
        # Detailed results
        st.markdown("## 📝 Interview Summary")
        
        for i, conv in enumerate(st.session_state.conversation_history):
            with st.expander(f"Question {conv['question_number']}{' (Follow-up)' if conv.get('is_follow_up') else ''} - Score: {conv.get('score', 'N/A')}/10"):
                st.markdown(f"**Question:** {conv['question']}")
                st.markdown(f"**Your Answer:** {conv['answer']}")
                if 'feedback' in conv:
                    st.markdown(f"**Feedback:** {conv['feedback']}")
        
        # Download results
        st.markdown("## 💾 Download Results")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Download as JSON"):
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
                    label="⬇️ Download JSON",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"interview_report_{role.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📄 Download as Text Summary"):
                # Create a text summary
                summary = f"AI Interview Simulator Report\n"
                summary += f"===========================\n\n"
                summary += f"Candidate: {st.session_state.user_profile.get('name', 'Anonymous')}\n"
                summary += f"Role: {role} ({level})\n"
                summary += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                summary += f"Overall Score: {avg_score:.1f}/10\n\n"
                
                for i, conv in enumerate(st.session_state.conversation_history):
                    summary += f"Question {conv['question_number']}{' (Follow-up)' if conv.get('is_follow_up') else ''}:\n"
                    summary += f"{conv['question']}\n\n"
                    summary += f"Your Answer:\n{conv['answer']}\n\n"
                    if 'feedback' in conv:
                        summary += f"Feedback:\n{conv['feedback']}\n\n"
                    summary += "---\n\n"
                
                st.download_button(
                    label="⬇️ Download Text Report",
                    data=summary,
                    file_name=f"interview_summary_{role.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
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
            st.session_state.use_fallback_mode = False
            st.session_state.follow_up_mode = False
            st.session_state.follow_up_count = 0
            st.rerun()

# Footer
st.markdown("---")
st.markdown("**💡 Interview Tips:**")
st.markdown("""
- **Use the STAR method** (Situation, Task, Action, Result) for behavioral questions
- **Give specific examples** from your experience
- **Be concise but thorough** - aim for 1-2 minute answers
- **Show your problem-solving process** - how you think is as important as what you know
- **Ask clarifying questions** if needed
- **Practice aloud** to improve your delivery
""")

# Student-specific tips
if st.session_state.get('current_level') == "Student" or (not st.session_state.interview_started and st.session_state.get('user_profile', {}).get('level') == "Student"):
    st.markdown("**🎓 Student-Specific Tips:**")
    st.markdown("""
    - **Highlight academic projects** and what you learned from them
    - **Discuss relevant coursework** and how it applies to the role
    - **Mention extracurricular activities** that demonstrate relevant skills
    - **Be honest about what you don't know** but show eagerness to learn
    - **Research the company** and connect your skills to their needs
    """)
