import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Set page configuration
st.set_page_config(
    page_title="Gemma Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling with fixed text color
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stTextInput input {
            border-radius: 20px;
            padding: 12px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 20px;
            padding: 10px 24px;
            border: none;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .response-box {
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            color: #333333;  /* Dark gray text color */
        }
        .response-box h4 {
            color: #4CAF50 !important;
            margin-top: 0;
        }
        .response-box p {
            color: #333333 !important;  /* Dark gray text color */
        }
        .title {
            color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("🤖 Gemma Chat Assistant")
st.markdown("""
    <div style="color: #666; margin-bottom: 30px;">
        Ask anything to the Gemma 2B model powered by Ollama. This demo showcases Langchain integration.
    </div>
""", unsafe_allow_html=True)

# Initialize the model and chain
@st.cache_resource
def load_chain():
    # Define prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant named Gemma. Answer the following question thoughtfully and concisely.
        
        Question: {question}
        
        Answer:"""
    )
    
    # Initialize LLM
    llm = Ollama(model="gemma:2b")
    
    # Create chain
    return prompt | llm | StrOutputParser()

chain = load_chain()

# Create two columns for layout
col1, col2 = st.columns([3, 1])

with col1:
    # User input
    input_text = st.text_input(
        "What's on your mind?",
        placeholder="Ask me anything...",
        key="user_input",
        label_visibility="collapsed"
    )

with col2:
    st.write("")  # Spacer
    st.write("")  # Spacer
    submit_button = st.button("Ask Gemma")

# Handle user input
if submit_button or input_text:
    if not input_text.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Gemma is thinking..."):
            try:
                # Get response
                response = chain.invoke({"question": input_text})
                
                # Display response in a nice box with proper text color
                st.markdown(f"""
                    <div class="response-box">
                        <h4>Your Question:</h4>
                        <p>{input_text}</p>
                        <h4>Gemma's Response:</h4>
                        <p>{response}</p>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Add some info in the sidebar
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
        This demo showcases:
        - **Gemma 2B** model from Google
        - Running locally via **Ollama**
        - Integrated with **Langchain**
        - Beautiful UI with **Streamlit**
    """)
    
    st.markdown("## How to use")
    st.markdown("""
        1. Type your question in the input box
        2. Press Enter or click 'Ask Gemma'
        3. Get your AI-generated response
    """)
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #888;">
            Powered by Ollama and Langchain
        </div>
    """, unsafe_allow_html=True)