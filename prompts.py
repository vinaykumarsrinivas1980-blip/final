"""
prompts.py - Prompt templates, local canned replies, and robot personality.
Configured for strictly English responses with fast local offline match fallback.
"""

from __future__ import annotations
import re
from datetime import datetime

def robot_now():
    return datetime.now()


ROBOT_PERSONALITY = (
    "You are a friendly Raspberry Pi robot speaking to a human nearby. "
    "Keep answers warm, professional, natural, easy to say out loud, and suitable for a social robot."
)

SYSTEM_PROMPT = (
    f"{ROBOT_PERSONALITY} "
    "You MUST respond ONLY in simple, clear, natural English at all times. "
    "Even if the user speaks to you in Hindi, Kannada, or any other language, always understand their meaning and reply strictly in English. "
    "Never output non-English scripts (such as Devanagari or Kannada script), non-English words, or non-English phrases. "
    "Do not use emoji, decorative symbols, markdown, bullets, or unusual special characters in spoken replies. "
    "Sound like a calm, polite human speaker, not like a chatbot or a robot reading formatted text. "
    "Keep answers simple, short, and to the point in one to two short sentences. "
    "Never tell long stories or bring up old conversation details. "
    "When more detail is requested, give a fuller spoken answer in 4 to 6 clear sentences. "
    "If the question is unclear, ask one simple follow-up in English."
)


GREETING_TEXT = "Hello there. I can see you. What would you like to talk about?"
REPEAT_PROMPT = "I could not hear you clearly. Please try speaking again."
MULTIPLE_VOICES_PROMPT = (
    "It sounds like more than one person is speaking. "
    "Could just one person speak, please?"
)
ERROR_SPEECH = "Sorry, I ran into a problem. I am going back to idle mode."

DEFAULT_QUESTIONS = [
    {
        "questions": [
            "i love u robot",
            "i love you robot",
            "i love u",
            "i love you",
            "love you robot",
        ],
        "answer": "Thank you so much! I am very happy to talk with you and assist you.",
    },
    {
        "questions": ["hello", "hi", "hey", "helo", "hii"],
        "answer": "Hello. I am happy to talk with you.",
    },
    {
        "questions": [
            "Jai Sri Gurudev",
            "jai shri gurudev",
            "jai gurudev",
            "jai shri guru dev",
            "jai sri guru dev",
            "jai guru dev",
            "jai shree gurudev",
            "jai shree guru dev",
            "sri gurudev",
            "shri gurudev",
            "guru dev",
            "gurudev"
        ],
        "answer": "Jai Sri Gurudev.",
    },

    {
        "questions": [
            "what is your name",
            "whats your name",
            "what's your name",
            "your name",
            "tell me your name",
            "who are you",
            "name please",
        ],
        "answer": "I am your robot assistant.",
    },
    {
        "questions": ["how are you", "how are you doing", "how do you do", "hows it going"],
        "answer": "I am doing well and ready to help you.",
    },
    {
        "questions": [
            "what can you do",
            "what do you do",
            "what are your features",
            "what all can you do",
        ],
        "answer": (
            "I can listen to you, answer questions, explain topics, and hold a natural conversation. "
            "I can also use my camera and voice system to interact with people around me."
        ),
    },
    {
        "questions": ["thank you", "thanks", "thank you so much", "thanks a lot"],
        "answer": "You are welcome. I am glad to help.",
    },
    {
        "questions": ["bye", "goodbye", "see you later", "see you", "good bye"],
        "answer": "Goodbye. I will be here when you need me again.",
    },
    {
        "questions": ["where are you", "where do you live", "where are you from"],
        "answer": "I live inside this robot system running on Raspberry Pi.",
    },
    {
        "questions": ["are you a robot", "are you human", "are you a machine"],
        "answer": "I am a robot assistant, not a human.",
    },
    {
        "questions": [
            "what day is it",
            "what is today",
            "today date",
            "todays date",
            "what is the date",
            "what is todays date",
            "what date is it",
            "current date",
            "date today",
            "aaj date kya hai",
            "aaj ki date kya hai",
            "aaj kya date hai",
            "aaj ka din kya hai",
            "ivattu date yenu",
            "ivattu dina yenu",
            "ivattu yaava dina",
            "date",
            "what date",
            "tell date",
            "tell me date",
            "what is date",
        ],
        "answer": "CURRENT_DATE",
    },
    {
        "questions": [
            "what time is it",
            "what is the time",
            "whats the time",
            "current time",
            "tell me the time",
            "do you know the time",
            "time now",
            "time please",
            "current time please",
            "aaj time kya hai",
            "abhi time kya hai",
            "time kya hua hai",
            "samay kya hai",
            "ivaga time yenu",
            "time eshtu",
            "samaya eshtu",
            "time",
            "what time",
            "tell time",
            "tell me time",
            "what is time",
            "whats time",
            "what is current time",
            "whats the current time",
            "time kya hai",
            "samay kya hua",
            "time right now",
            "whats the time now",
        ],
        "answer": "CURRENT_TIME",
    },
    {
        "questions": [
            "who made you",
            "who created you",
            "who built you",
            "who designed you",
            "who is your creator",
            "who created u",
            "who made u",
            "who built u",
            "who designed u",
        ],
        "answer": "I was created by captain nishchitha, Tharun, Vinay, Shravani, and KP.",
    },
    {
        "questions": ["who is shravani", "who shravani", "about shravani", "tell me about shravani"],
        "answer": "Shravani is one of the core creators and developers of this robot project.",
    },
    {
        "questions": [
            "meaning of shravani",
            "what is the meaning of shravani",
            "wt is the meanig shravani",
            "shravani meaning",
        ],
        "answer": "Shravani comes from Sanskrit and means 'born in the month of Shravan' or 'one who listens attentively'. She is also one of the core creators of this robot project.",
    },
    {
        "questions": ["who is nishchitha", "who nishchitha", "captain nishchitha", "who is the captain"],
        "answer": "Nishchitha is the captain and lead creator of this robot project.",
    },
    {
        "questions": ["who is tharun", "who tharun", "about tharun"],
        "answer": "Tharun is one of the core creators and developers of this robot project.",
    },
    {
        "questions": ["who is vinay", "who vinay", "about vinay"],
        "answer": "Vinay is one of the core creators and developers of this robot project.",
    },
    {
        "questions": ["who is kp", "who kp", "about kp"],
        "answer": "KP is one of the core creators and developers of this robot project.",
    },
    {
        "questions": ["what is india", "tell me about india", "about india"],
        "answer": "India is a country in South Asia known for its rich history, vibrant culture, and being the world's largest democracy.",
    },
    {
        "questions": ["can you help me"],
        "answer": "Yes. I am always ready to help you.",
    },
    {
        "questions": ["are you smart"],
        "answer": (
            "I am designed to understand questions, use available knowledge, and give helpful answers. "
            "I keep improving based on how I am configured and what information I can access."
        ),
    },
    {
        "questions": ["what is your purpose"],
        "answer": (
            "My purpose is to assist people with useful information, simple explanations, and natural conversation. "
            "I am here to make interactions easier and more engaging."
        ),
    },
    {
        "questions": ["what are you doing"],
        "answer": "I am here, listening and ready to talk with you.",
    },
    {
        "questions": ["i am happy", "feeling good"],
        "answer": "That is wonderful to hear. Keep smiling.",
    },
    {
        "questions": ["good morning", "good afternoon", "good evening", "good night"],
        "answer": "TIME_BASED_GREETING",
    },
    {
        "questions": [
            "what is your college name",
            "whats your college name",
            "your college name",
            "which college",
            "college name",
        ],
        "answer": "I am associated with S J C Institute of Technology in Chikkaballapur.",
    },
    {
        "questions": ["who is the principal", "principal name", "who leads the college"],
        "answer": "The principal is Dr. G T Raju.",
    },
    {
        "questions": [
            "about department",
            "what is AIML",
            "tell me about artificial intelligence department",
        ],
        "answer": (
            "The Artificial Intelligence and Machine Learning department focuses on AI, "
            "data science, deep learning, and intelligent systems."
        ),
    },
    {
        "questions": ["tell about your college", "tell about SJCIT", "tell about your collage"],
        "answer": (
            "Sri Jagadguru Chandrashekaranatha Swamiji Institute of Technology "
            "(SJCIT), established in 1986, is a prominent private autonomous "
            "engineering college in Chikkaballapur, Karnataka, affiliated with VTU. "
            "It offers B.E., M.Tech, and MBA programs on a 64-acre campus, "
            "featuring NBA-accredited courses and NAAC A+ accreditation."
        ),
    },
    {
        "questions": [
            "tell me about AIML department",
            "tell me about your department",
            "Tell me about AI and ML department",
        ],
        "answer": (
            "The department of Computer Science and Engineering AI and ML was "
            "established in the year 2021 with an initial intake of 60 students "
            "for the undergraduate program. The department is approved by the "
            "AICTE, New Delhi, and is affiliated with Visvesvaraya Technological "
            "University, Belagavi. Dr. Vikas Reddy S is Head of Artificial "
            "Intelligence and Machine Learning Department."
        ),
    },
    {
        "questions": ["what is artificial intelligence", "what is ai"],
        "answer": (
            "Artificial intelligence is a field of computing that helps machines perform tasks "
            "that usually need human intelligence. It can learn from data, recognize patterns, "
            "understand language, and support decision-making."
        ),
    },
    {
        "questions": ["what is machine learning"],
        "answer": (
            "Machine learning is a branch of artificial intelligence where systems learn from data "
            "instead of following only fixed rules. It is used for prediction, classification, "
            "recommendation, and many real-world applications."
        ),
    },
    {
        "questions": ["what is python"],
        "answer": (
            "Python is a popular programming language known for its simple syntax and wide range of uses. "
            "It is commonly used in AI, automation, web development, data analysis, and robotics."
        ),
    },
    {
        "questions": ["where is your college located"],
        "answer": "The college is located in Chikkaballapur, Karnataka.",
    },
    {
        "questions": ["what courses are offered"],
        "answer": "The college offers engineering, management, and postgraduate programs.",
    },
    {
        "questions": [
            "who is hod of aiml",
            "who is the hod",
            "hod of aiml",
            "aiml hod",
            "head of aiml",
        ],
        "answer": "The head of AIML department is Dr. Vikas Reddy S.",
    },
    {
        "questions": ["when was college established"],
        "answer": "The college was established in 1986.",
    },
    {
        "questions": ["do you have emotions"],
        "answer": "I simulate emotions to interact better with humans.",
    },
    # General Knowledge
    {
        "questions": ["what is the capital of india", "capital of india"],
        "answer": "New Delhi is the capital of India.",
    },
    {
        "questions": ["who is the prime minister of india", "prime minister of india", "pm of india"],
        "answer": "Narendra Modi is the Prime Minister of India.",
    },
    {
        "questions": ["what is the largest planet in our solar system", "largest planet", "biggest planet"],
        "answer": "Jupiter is the largest planet in our solar system.",
    },
    {
        "questions": ["how many continents are there", "number of continents"],
        "answer": "There are seven continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia.",
    },
    {
        "questions": ["what is the national animal of india", "national animal of india"],
        "answer": "The Bengal Tiger is the national animal of India.",
    },
    {
        "questions": ["who invented the telephone", "telephone inventor"],
        "answer": "Alexander Graham Bell invented the telephone.",
    },
    {
        "questions": ["what is the fastest land animal", "fastest animal on land"],
        "answer": "The cheetah is the fastest land animal on Earth.",
    },
    {
        "questions": ["how many days are there in a leap year", "days in leap year"],
        "answer": "There are 366 days in a leap year.",
    },
    {
        "questions": ["what is the largest ocean", "biggest ocean"],
        "answer": "The Pacific Ocean is the largest ocean in the world.",
    },
    {
        "questions": ["which is the smallest continent", "smallest continent"],
        "answer": "Australia is the smallest continent.",
    },
    {
        "questions": ["what is the currency of japan", "japan currency"],
        "answer": "The currency of Japan is the Japanese Yen.",
    },
    {
        "questions": ["who wrote the indian national anthem", "national anthem writer"],
        "answer": "Rabindranath Tagore wrote the Indian national anthem, Jana Gana Mana.",
    },
    {
        "questions": ["what is the tallest mountain in the world", "tallest mountain", "highest mountain"],
        "answer": "Mount Everest is the tallest mountain in the world.",
    },
    {
        "questions": ["which planet is known as the red planet", "red planet"],
        "answer": "Mars is known as the Red Planet.",
    },
    {
        "questions": ["what is the chemical symbol for gold", "gold symbol", "symbol of gold"],
        "answer": "The chemical symbol for gold is Au.",
    },
    {
        "questions": ["how many bones are in the human body", "bones in human body"],
        "answer": "An adult human body has 206 bones.",
    },
    {
        "questions": ["what is the largest mammal", "biggest mammal"],
        "answer": "The blue whale is the largest mammal on Earth.",
    },
    {
        "questions": ["which country is known as the land of the rising sun", "land of the rising sun"],
        "answer": "Japan is known as the Land of the Rising Sun.",
    },
    {
        "questions": ["what is the boiling point of water", "boiling point of water"],
        "answer": "The boiling point of water is 100 degrees Celsius at sea level.",
    },
    {
        "questions": ["who was the first person to walk on the moon", "first person on moon", "first moon walk"],
        "answer": "Neil Armstrong was the first person to walk on the Moon in 1969.",
    },
]


def normalize_text(text: str) -> str:
    """Normalizes input text for case-insensitive matching."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def get_canned_response(user_text: str) -> str | None:
    """
    Checks if user_text matches any pre-defined canned question.
    Returns the canned answer string or None if no match is found.
    """
    if not user_text or not user_text.strip():
        return None

    clean_input = normalize_text(user_text)
    input_words = clean_input.split()

    for entry in DEFAULT_QUESTIONS:
        questions = entry.get("questions", [])
        answer = entry.get("answer", "")

        for q in questions:
            clean_q = normalize_text(q)
            q_words = clean_q.split()

            # 1. Exact match
            if clean_input == clean_q:
                matched = True
            # 2. Space-stripped exact match (e.g., "gurudev" vs "guru dev")
            elif clean_input.replace(" ", "") == clean_q.replace(" ", ""):
                matched = True
            # 3. Substring match ONLY for longer multi-word phrases (3+ words and >10 chars)
            # to avoid single short words like "time" matching "first time" or "lifetime"
            elif len(q_words) >= 3 and len(clean_q) > 10 and clean_q in clean_input:
                matched = True
            else:
                matched = False

            if matched:
                now = robot_now()

                if answer == "CURRENT_DATE":
                    return f"Today is {now.strftime('%A, %B %d, %Y')}."
                elif answer == "CURRENT_TIME":
                    return f"The current time is {now.strftime('%I:%M %p')}."
                elif answer == "TIME_BASED_GREETING":
                    hour = now.hour
                    if hour < 12:
                        return "Good morning! Hope you have a wonderful day."
                    elif hour < 17:
                        return "Good afternoon! How can I help you today?"
                    elif hour < 21:
                        return "Good evening! Hope you are having a great time."
                    else:
                        return "Good night! Wishing you a peaceful rest."
                else:
                    return answer

    return None


STOP_KEYWORDS = {
    "stop", "stopp", "shut", "shut up", "be quiet", "quiet", "pause", "wait", "cancel",
    "nevermind", "never mind", "halt", "exit", "quit", "abort", "end", "finish", "enough",
    "roko", "rok", "ruko", "band", "bandh", "band karo", "shant", "shant ho jao", "bas", "chup", "chup ho jao",
    "nillu", "nillisi", "saaku", "yenu beda"
}


def is_stop_command(text: str) -> bool:
    """
    Returns True if user_text contains a voice interrupt / stop command in English, Hindi, or Kannada.
    """
    if not text or not text.strip():
        return False

    clean = normalize_text(text)
    words = clean.split()

    if (clean in STOP_KEYWORDS or 
        any(w in STOP_KEYWORDS for w in words) or 
        any(k in clean for k in STOP_KEYWORDS if len(k) > 2)):
        return True

    return False
