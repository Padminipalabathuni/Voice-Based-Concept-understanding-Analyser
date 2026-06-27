from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

model = SentenceTransformer('all-MiniLM-L6-v2')

CONCEPT_REFERENCES = {
    "Machine Learning": """Machine learning is a subset of artificial intelligence 
    that enables systems to learn and improve from experience without being explicitly 
    programmed. It focuses on developing computer programs that can access data and 
    use it to learn for themselves.""",

    "Cloud Computing": """Cloud computing is the delivery of computing services 
    including servers, storage, databases, networking, software, analytics and 
    intelligence over the internet to offer faster innovation, flexible resources 
    and economies of scale.""",

    "Deep Learning": """Deep learning is a subset of machine learning that uses 
    neural networks with many layers to learn representations of data with multiple 
    levels of abstraction.""",

    "Artificial Intelligence": """Artificial intelligence is the simulation of 
    human intelligence processes by machines, especially computer systems, including 
    learning, reasoning, problem solving, perception and language understanding.""",

    "Neural Networks": """Neural networks are computing systems inspired by biological 
    neural networks consisting of interconnected nodes that process information using 
    connectionist approaches to computation.""",

    "Data Science": """Data science is an interdisciplinary field that uses scientific 
    methods, processes, algorithms and systems to extract knowledge and insights 
    from structured and unstructured data.""",

    "Natural Language Processing": """Natural language processing is a branch of 
    artificial intelligence that helps computers understand, interpret and manipulate 
    human language bridging the gap between human communication and computer understanding.""",

    "Computer Vision": """Computer vision is a field of artificial intelligence that 
    trains computers to interpret and understand the visual world using digital images 
    and deep learning models to accurately identify and classify objects."""
}

def semantic_similarity(text1: str, text2: str) -> float:
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return round(float(similarity), 4)

def auto_detect_topic(text: str, api_key: str) -> str:
    try:
        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel('gemini-1.5-flash')
        topics = list(CONCEPT_REFERENCES.keys())
        prompt = f"""Given this spoken explanation: "{text}"
Which topic from this list best matches?
Topics: {', '.join(topics)}
Reply with ONLY the topic name."""
        response = gemini.generate_content(prompt)
        detected = response.text.strip()
        return detected if detected in CONCEPT_REFERENCES else topics[0]
    except:
        return "Machine Learning"

def get_gemini_feedback(text: str, topic: str, score: float, api_key: str) -> str:
    try:
        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""A student explained "{topic}" and got score {score}/100.
Their explanation: "{text}"
Give 3-4 lines of friendly feedback on what they did well and what to improve."""
        response = gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Could not generate feedback: {str(e)}"

def get_reference(topic: str) -> str:
    return CONCEPT_REFERENCES.get(topic, "Topic not found")
