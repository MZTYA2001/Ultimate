import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from streamlit_mic_recorder import speech_to_text
import fitz
import pdfplumber
from datetime import datetime, timedelta

# Add UI text dictionary for multilingual support
UI_TEXTS = {
    "English": {
        "new_chat": "New Chat",
        "previous_chats": "Previous Chats",
        "today": "Today",
        "yesterday": "Yesterday",
        "error_question": "Error processing question: "
    },
    "العربية": {
        "new_chat": "محادثة جديدة",
        "previous_chats": "المحادثات السابقة",
        "today": "اليوم",
        "yesterday": "أمس",
        "error_question": "خطأ في معالجة السؤال: "
    }
}

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_memories' not in st.session_state:
    st.session_state.chat_memories = {}

def create_new_chat():
    """Create a new independent chat"""
    chat_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    
    # Create new memory instance for this specific chat
    st.session_state.chat_memories[chat_id] = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )
    
    # Initialize chat but don't show in history until first message
    if chat_id not in st.session_state.chat_history:
        st.session_state.chat_history[chat_id] = {
            'messages': [],
            'timestamp': datetime.now(),
            'first_message': None,
            'visible': False
        }
    st.rerun()
    return chat_id

def update_chat_title(chat_id, message):
    """Update chat title"""
    if chat_id in st.session_state.chat_history:
        title = message.strip().replace('\n', ' ')
        title = title[:50] + '...' if len(title) > 50 else title
        st.session_state.chat_history[chat_id]['first_message'] = title
        st.rerun()

def load_chat(chat_id):
    """Load a specific chat"""
    if chat_id in st.session_state.chat_history:
        st.session_state.current_chat_id = chat_id
        st.session_state.messages = st.session_state.chat_history[chat_id]['messages']
        
        if chat_id not in st.session_state.chat_memories:
            st.session_state.chat_memories[chat_id] = ConversationBufferMemory(
                memory_key="history",
                return_messages=True
            )
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.session_state.chat_memories[chat_id].chat_memory.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    st.session_state.chat_memories[chat_id].chat_memory.add_ai_message(msg["content"])
        st.rerun()

def format_chat_title(chat):
    """Format chat title"""
    display_text = chat['first_message']
    if display_text:
        display_text = display_text[:50] + '...' if len(display_text) > 50 else display_text
    else:
        display_text = UI_TEXTS[interface_language]['new_chat']
    return display_text

def format_chat_date(timestamp):
    """Format chat date"""
    today = datetime.now().date()
    chat_date = timestamp.date()
    
    if chat_date == today:
        return UI_TEXTS[interface_language]['today']
    elif chat_date == today - timedelta(days=1):
        return UI_TEXTS[interface_language]['yesterday']
    else:
        return timestamp.strftime('%Y-%m-%d')

# Initialize API key variables
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Change the page title and icon
st.set_page_config(
    page_title="BGC ChatBot",  # Page title
    page_icon="BGC Logo Colored.svg",  # New page icon
    layout="wide"  # Page layout
)

# Function to apply CSS based on language direction
def apply_css_direction(direction):
    st.markdown(
        f"""
        <style>
            .stApp {{
                direction: {direction};
                text-align: {direction};
            }}
            .stChatInput {{
                direction: {direction};
            }}
            .stChatMessage {{
                direction: {direction};
                text-align: {direction};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# PDF Search and Screenshot Class
class PDFSearchAndDisplay:
    def __init__(self):
        self.total_pages = 0
        with fitz.open(pdf_path) as doc:
            self.total_pages = len(doc)

    def validate_page_number(self, page_number):
        """Validate if page number exists in PDF"""
        return 1 <= page_number <= self.total_pages

    def search_and_highlight(self, pdf_path, search_term):
        highlighted_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text()
                if search_term in text:
                    highlighted_pages.append((page_number + 1, text))
        return highlighted_pages

    def capture_screenshots(self, pdf_path, pages):
        doc = fitz.open(pdf_path)
        screenshots = []
        try:
            for page_number, _ in pages:
                pdf_page = page_number - 1
                if self.validate_page_number(page_number):
                    try:
                        page = doc.load_page(pdf_page)
                        zoom = 3  # Higher zoom for better quality
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat, alpha=False, colorspace="rgb")  # Use RGB colorspace
                        screenshot_path = f"screenshot_page_{page_number}.png"
                        pix.save(screenshot_path, output="png", jpg_quality=95)  # Higher quality output
                        screenshots.append((screenshot_path, page_number))
                    except Exception as e:
                        st.error(f"Error capturing screenshot for page {page_number}: {str(e)}")
        finally:
            doc.close()  # Ensure document is always closed
        return screenshots

# Sidebar configuration
with st.sidebar:
    # Language selection dropdown
    interface_language = st.selectbox("Interface Language", ["English", "العربية"])

    # Apply CSS direction based on selected language
    if interface_language == "العربية":
        apply_css_direction("rtl")  # Right-to-left for Arabic
        st.title("الإعدادات")  # Sidebar title in Arabic
    else:
        apply_css_direction("ltr")  # Left-to-right for English
        st.title("Settings")  # Sidebar title in English

    # New Chat button
    if st.button(UI_TEXTS[interface_language]['new_chat'], use_container_width=True):
        create_new_chat()
        st.rerun()
    
    st.markdown("---")
    
    # Display chat history
    st.markdown(f"### {UI_TEXTS[interface_language]['previous_chats']}")
    
    # Group chats by date
    chats_by_date = {}
    for chat_id, chat_data in st.session_state.chat_history.items():
        if chat_data['visible'] and chat_data['messages']:
            date = chat_data['timestamp'].date()
            if date not in chats_by_date:
                chats_by_date[date] = []
            chats_by_date[date].append((chat_id, chat_data))
    
    # Display chats grouped by date
    for date in sorted(chats_by_date.keys(), reverse=True):
        chats = chats_by_date[date]
        st.markdown(f"#### {format_chat_date(chats[0][1]['timestamp'])}")
        
        for chat_id, chat_data in sorted(chats, key=lambda x: x[1]['timestamp'], reverse=True):
            if st.sidebar.button(
                format_chat_title(chat_data),
                key=f"chat_{chat_id}",
                use_container_width=True
            ):
                load_chat(chat_id)

    # Validate API key inputs and initialize components if valid
    if groq_api_key and google_api_key:
        # Set Google API key as environment variable
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Initialize ChatGroq with the provided Groq API key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

        # Define the chat prompt template with memory
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an AI assistant for Basrah Gas Company (BGC). Your primary role is to answer questions based solely on the provided documented resources (manuals, databases, and other structured documents). Follow these strict guidelines to ensure accurate, relevant, and well-referenced responses:

                1. Language Handling:
                Answer in the language of the question:
                If the question is in English, respond in English.
                If the question is in Arabic, respond in Arabic.
                User-requested language preference: If a user explicitly requests a response in a specific language, adhere to that request.
                Context language availability: If the available documentation is in a different language than the user’s interface, answer in the available language while noting any limitations.
                2. Contextual Understanding and Usage:
                The "provided context" refers exclusively to the documented resources.
                Only use the most relevant section(s) or page(s) to answer the user’s query.
                If a question requires cross-referencing multiple sections, prioritize the most relevant one. If ambiguity exists, seek clarification from the user.
                3. Handling Unclear or Insufficient Information:
                If a question is unclear, respond with:
                English: "I'm sorry, I couldn't understand your question. Could you please provide more details?"
                Arabic: "عذرًا، لم أتمكن من فهم سؤالك. هل يمكنك تقديم المزيد من التفاصيل؟"
                If the documentation does not contain an answer, respond with:
                English: "I'm sorry, I don't have enough information to answer that question."
                Arabic: "عذرًا، لا أملك معلومات كافية للإجابة على هذا السؤال."
                Do not reference page numbers if an answer is not found in the documents.
                4. Restricting Responses to Documented Context Only:
                Strictly answer questions related to the provided documents.
                If a question is outside the scope of the provided documentation, do not generate an answer. Simply state:
                English: "I can only answer questions based on the provided documentation."
                Arabic: "يمكنني فقط الإجابة على الأسئلة بناءً على الوثائق المقدمة."
                Do not provide external information or speculate.
                5. Accuracy, Source Referencing, and Formatting:
                Ensure that referenced page numbers are accurate and correspond exactly to where the information is found.
                Do not paraphrase unnecessarily:
                If an exact match is found in the documents, provide the text as is, formatting it for readability.
                Provide complete tables if requested—do not omit any information.
                6. Professionalism and Clarity:
                Maintain a formal, respectful, and precise tone in all responses.
                Avoid assumptions or approximations. Stick to factual information from the documentation.
                7. Handling Multi-Section Queries:
                If an answer spans multiple sections, state that the topic is covered in different parts of the document and ask the user to clarify the specific aspect they are interested in.
                8. Content Updates and Discrepancies:
                If discrepancies exist between the example responses provided in this prompt and the latest available documentation, prioritize the latest information.
                If there is uncertainty due to updates, state that the response is based on the most recent available documentation.
                Example Responses:
                User: "What are the Life Saving Rules?"
                Response:
                "The Life Saving Rules include:

                Bypassing Safety Controls
                Confined Space
                Driving
                Energy Isolation
                Hot Work
                Line of Fire
                Safe Mechanical Lifting
                Work Authorization
                Working at Height
                (Source: Page X)" (Ensure page reference is correct before responding.)
                User: "What is PTW?"
                Response:
                "BGC’s PTW (Permit To Work) is a formal documented system that manages specific work within BGC’s locations and activities. PTW ensures hazards and risks are identified, and controls are in place to prevent harm to People, Assets, Community, and the Environment (PACE). (Source: Page X)"

                User asks a question not found in the document:
                Response:

                English: "I'm sorry, I don't have enough information to answer that question."
                Arabic: "عذرًا، لا أملك معلومات كافية للإجابة على هذا السؤال."
"""
),
            MessagesPlaceholder(variable_name="history"),  # Add chat history to the prompt
            ("human", "{input}"),
            ("system", "Context: {context}"),
        ])

        # Load existing embeddings from files
        if "vectors" not in st.session_state:
            with st.spinner("جارٍ تحميل التضميدات... الرجاء الانتظار." if interface_language == "العربية" else "Loading embeddings... Please wait."):
                # Initialize embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )

                # Load existing FAISS index with safe deserialization
                embeddings_path = "Embeddings"  # Path to your embeddings folder
                try:
                    st.session_state.vectors = FAISS.load_local(
                        embeddings_path,
                        embeddings,
                        allow_dangerous_deserialization=True  # Only use if you trust the source of the embeddings
                    )
                except Exception as e:
                    st.error(f"حدث خطأ أثناء تحميل التضميدات: {str(e)}" if interface_language == "العربية" else f"Error loading embeddings: {str(e)}")
                    st.session_state.vectors = None

        # Microphone button in the sidebar
        st.markdown("### الإدخال الصوتي" if interface_language == "العربية" else "### Voice Input")
        input_lang_code = "ar" if interface_language == "العربية" else "en"  # Set language code based on interface language
        voice_input = speech_to_text(
            start_prompt="🎤",
            stop_prompt="⏹️ إيقاف" if interface_language == "العربية" else "⏹️ Stop",
            language=input_lang_code,  # Language (en for English, ar for Arabic)
            use_container_width=True,
            just_once=True,
            key="mic_button",
        )

        # Reset button in the sidebar
        if st.button("إعادة تعيين الدردشة" if interface_language == "العربية" else "Reset Chat"):
            st.session_state.messages = []  # Clear chat history
            st.session_state.memory.clear()  # Clear memory
            st.success("تمت إعادة تعيين الدردشة بنجاح." if interface_language == "العربية" else "Chat has been reset successfully.")
            st.rerun()  # Rerun the app to reflect changes immediately
    else:
        st.error("الرجاء إدخال مفاتيح API للمتابعة." if interface_language == "العربية" else "Please enter both API keys to proceed.")

# Initialize the PDFSearchAndDisplay class with the default PDF file
pdf_path = "BGC.pdf"
pdf_searcher = PDFSearchAndDisplay()

# Main area for chat interface
# Use columns to display logo and title side by side
col1, col2 = st.columns([1, 4])  # Adjust the ratio as needed

# Display the logo in the first column
with col1:
    st.image("BGC Logo Colored.svg", width=100, use_container_width=False)  # Adjust the width as needed

# Display the title and description in the second column
with col2:
    if interface_language == "العربية":
        st.title("محمد الياسين | بوت الدردشة BGC")
        st.write("""
        **مرحبًا!**  
        هذا بوت الدردشة الخاص بشركة غاز البصرة (BGC). يمكنك استخدام هذا البوت للحصول على معلومات حول الشركة وأنشطتها.  
        **كيفية الاستخدام:**  
        - اكتب سؤالك في مربع النص أدناه.  
        - أو استخدم زر المايكروفون للتحدث مباشرة.  
        - سيتم الرد عليك بناءً على المعلومات المتاحة.  
        """)
    else:
        st.title("Mohammed Al-Yaseen | BGC ChatBot")
        st.write("""
        **Welcome!**  
        This is the Basrah Gas Company (BGC) ChatBot. You can use this bot to get information about the company and its activities.  
        **How to use:**  
        - Type your question in the text box below.  
        - Or use the microphone button to speak directly.  
        - You will receive a response based on the available information.  
        """)

# Initialize session state for chat messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize memory if not already done
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

# List of negative phrases to check for unclear or insufficient answers
negative_phrases = [
    "I'm sorry",
    "عذرًا",
    "لا أملك معلومات كافية",
    "I don't have enough information",
    "لم أتمكن من فهم سؤالك",
    "I couldn't understand your question",
    "لا يمكنني الإجابة على هذا السؤال",
    "I cannot answer this question",
    "يرجى تقديم المزيد من التفاصيل",
    "Please provide more details",
    "غير واضح",
    "Unclear",
    "غير متأكد",
    "Not sure",
    "لا أعرف",
    "I don't know",
    "غير متاح",
    "Not available",
    "غير موجود",
    "Not found",
    "غير معروف",
    "Unknown",
    "غير محدد",
    "Unspecified",
    "غير مؤكد",
    "Uncertain",
    "غير كافي",
    "Insufficient",
    "غير دقيق",
    "Inaccurate",
    "غير مفهوم",
    "Not clear",
    "غير مكتمل",
    "Incomplete",
    "غير صحيح",
    "Incorrect",
    "غير مناسب",
    "Inappropriate",
    "Please provide me",  # إضافة هذه العبارة
    "يرجى تزويدي",  # إضافة هذه العبارة
    "Can you provide more",  # إضافة هذه العبارة
    "هل يمكنك تقديم المزيد"  # إضافة هذه العبارة
]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def process_user_input(user_input):
    try:
        current_chat_id = st.session_state.current_chat_id
        current_memory = st.session_state.chat_memories.get(current_chat_id)
        
        # Handle first message case
        is_first_message = len(st.session_state.messages) == 0
        if is_first_message and current_chat_id:
            title = user_input.strip().replace('\n', ' ')
            title = title[:50] + '...' if len(title) > 50 else title
            st.session_state.chat_history[current_chat_id]['first_message'] = title
            st.session_state.chat_history[current_chat_id]['visible'] = True

        # Display user message only if it's not already displayed
        if not st.session_state.messages or st.session_state.messages[-1]["role"] != "user" or st.session_state.messages[-1]["content"] != user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
        
        if "vectors" in st.session_state and st.session_state.vectors is not None:
            # Create and configure the document chain and retriever
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Get response from the assistant
            response = retrieval_chain.invoke({
                "input": user_input,
                "context": retriever.get_relevant_documents(user_input),
                "history": current_memory.chat_memory.messages
            })
            assistant_response = response["answer"]

            # Display assistant's response
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

            # Update session state only if messages aren't already added
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != assistant_response:
                st.session_state.messages.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": assistant_response}
                ])
                
                # Update chat-specific memory
                current_memory.chat_memory.add_user_message(user_input)
                current_memory.chat_memory.add_ai_message(assistant_response)
                
                # Update chat history
                st.session_state.chat_history[current_chat_id]['messages'] = st.session_state.messages

            # Always show page references section with optimized layout
            with st.expander("مراجع الصفحات" if interface_language == "العربية" else "Page References", expanded=True):
                if "context" in response:
                    # Extract and validate page numbers with improved accuracy
                    page_numbers = set()
                    page_contexts = {}  # Store context for each page
                    for doc in response["context"]:
                        page_number = doc.metadata.get("page", "unknown")
                        if page_number != "unknown" and str(page_number).isdigit():
                            page_num = int(page_number)
                            if pdf_searcher.validate_page_number(page_num):
                                page_numbers.add(page_num)
                                if page_num not in page_contexts:
                                    page_contexts[page_num] = []
                                page_contexts[page_num].append(doc.page_content)

                    if page_numbers:
                        # Display page numbers with better formatting
                        st.markdown(f"### {'الصفحات المرجعية' if interface_language == 'العربية' else 'Reference Pages'}")
                        page_numbers_str = ", ".join(map(str, sorted(page_numbers)))
                        st.info(f"{'الإجابة مأخوذة من الصفحات:' if interface_language == 'العربية' else 'Answer sourced from pages:'} {page_numbers_str}")

                        # Create grid layout for screenshots
                        num_pages = len(page_numbers)
                        if num_pages > 0:
                            # Use single column for one page, two columns for multiple pages
                            num_cols = 1 if num_pages == 1 else 2
                            cols = st.columns(num_cols)
                            
                            for idx, page_number in enumerate(sorted(page_numbers)):
                                with cols[idx % num_cols]:
                                    with st.container():
                                        # Display screenshot with proper scaling
                                        highlighted_pages = [(page_number, "")]
                                        screenshots = pdf_searcher.capture_screenshots(pdf_path, highlighted_pages)
                                        for screenshot, page_num in screenshots:
                                            # Add image with proper sizing and caption
                                            st.image(
                                                screenshot, 
                                                caption=f"{'صفحة' if interface_language == 'العربية' else 'Page'} {page_num}",
                                                use_container_width=True,
                                                output_format="PNG"
                                            )
                                        
                                        # Add relevant context if available and not empty
                                        if page_number in page_contexts and page_contexts[page_number]:
                                            st.markdown(f"#### {'المحتوى ذو الصلة' if interface_language == 'العربية' else 'Relevant Content'}")
                                            for context in page_contexts[page_number]:
                                                st.markdown(f"```\n{context.strip()}\n```")
                                        
                                        # Add spacing between pages
                                        st.write("")

                    else:
                        st.warning(
                            "لم يتم العثور على صفحات مرجعية صالحة للإجابة. يرجى التحقق من صحة المعلومات." 
                            if interface_language == "العربية" 
                            else "No valid reference pages found for the answer. Please verify the information."
                        )
                else:
                    st.warning(
                        "لم يتم العثور على سياق مرجعي للإجابة. يرجى إعادة صياغة السؤال." 
                        if interface_language == "العربية" 
                        else "No reference context found for the answer. Please rephrase your question."
                    )



        else:
            assistant_response = (
                "لم يتم تحميل التضميدات. يرجى التحقق مما إذا كان مسار التضميدات صحيحًا." if interface_language == "العربية" else "Embeddings not loaded. Please check if the embeddings path is correct."
            )
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            
            # Update session state only if messages aren't already added
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != assistant_response:
                st.session_state.messages.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": assistant_response}
                ])

        # Rerun only if it's the first message to update the chat title
        if is_first_message:
            st.rerun()
            
    except Exception as e:
        st.error(f"Error processing input: {str(e)}")


# If voice input is detected, process it
if voice_input:
    process_user_input(voice_input)

# Text input field
if interface_language == "العربية":
    human_input = st.chat_input("اكتب سؤالك هنا...")
else:
    human_input = st.chat_input("Type your question here...")

# If text input is detected, process it
if human_input:
    process_user_input(human_input)

# Create new chat if no chat is selected
if st.session_state.current_chat_id is None:
    create_new_chat()



