import json
import requests
import os
import time
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import threading
from datetime import datetime
import logging
from collections import deque
import re
import webbrowser
import socket
import random
from balancer import format_balanced_equation
from typer import prompt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vahni.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ConversationManager:
    
    def __init__(self, max_history=20):
        self.history = deque(maxlen=max_history)
        self.session_start = datetime.now()
        self.message_count = 0
        self.topics_discussed = set()
        
    def add_interaction(self, query, response, metadata):
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'metadata': metadata
        })
        self.message_count += 1
        if metadata.get('topic'):
            self.topics_discussed.add(metadata['topic'])
    
    def get_context(self, n=3):
        return list(self.history)[-n:] if self.history else []
    
    def get_recent_topics(self):
        recent = []
        for item in reversed(self.history):
            topic = item.get('metadata', {}).get('topic')
            if topic and topic not in recent:
                recent.append(topic)
            if len(recent) >= 3:
                break
        return recent
    
    def clear(self):
        self.history.clear()
        self.message_count = 0
        self.topics_discussed.clear()
        self.session_start = datetime.now()
    
    def get_stats(self):
        return {
            'message_count': self.message_count,
            'topics_discussed': len(self.topics_discussed),
            'session_duration': str(datetime.now() - self.session_start),
            'recent_topics': list(self.topics_discussed)[-5:]
        }


class IntentClassifier:
    
    INTENT_PATTERNS = {
        'greeting': r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
        'gratitude': r'\b(thank|thanks|thx|ty|appreciate|grateful)\b',
        'farewell': r'\b(bye|goodbye|see you|later|gtg)\b',
        'clarification': r'\b(what|explain|clarify|elaborate|more detail|don\'t understand|didn\'t get)\b',
        'definition': r'\b(what is|what are|define|meaning of|definition)\b',
        'example': r'\b(example|give me|show me|demonstrate)\b',
        'comparison': r'\b(difference|compare|versus|vs|better)\b',
        'help': r'\b(help|assist|guide|how to)\b',
        'confirmation': r'\b(ok|okay|cool|nice|great|good|got it|understood|alright)\b'
    }
    
    @staticmethod
    def classify(query):
        query_lower = query.lower()
        intents = []
        
        for intent, pattern in IntentClassifier.INTENT_PATTERNS.items():
            if re.search(pattern, query_lower):
                intents.append(intent)
        
        return intents if intents else ['question']


class ResponseGenerator:
    
    def __init__(self, ollama_model="mistral"):
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.response_cache = {}
        
    def generate(self, prompt, temperature=0.3, max_tokens=200):
        """Generate response from Ollama"""
        cache_key = hash(prompt[:200])
        
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 2048,
                "num_thread": 4,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.2
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                result = self._clean_response(result)
                
                if len(result) > 20:
                    self.response_cache[cache_key] = result
                    if len(self.response_cache) > 50:
                        self.response_cache.pop(next(iter(self.response_cache)))
                return result
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return f"Error: Ollama returned status {response.status_code}"
        
        except requests.exceptions.Timeout:
            logger.error("Ollama timeout")
            return "I'm taking longer than expected. Try a simpler question or rephrase."
        except requests.exceptions.ConnectionError:
            logger.error("Ollama connection error")
            return "Cannot connect to the AI engine. Please ensure Ollama is running."
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"An error occurred: {type(e).__name__}"
    
    def _clean_response(self, text):
        lines = text.split('\n')
        seen = set()
        cleaned = []
        for line in lines:
            line_clean = line.strip()
            if line_clean and line_clean not in seen:
                seen.add(line_clean)
                cleaned.append(line)
        return '\n'.join(cleaned)
    
    def generate_greeting(self, name="there"):
        greetings = [
            f"Hi {name}! How can I help you with chemistry today?",
            f"Hello {name}! What chemistry topic would you like to explore?",
            f"Hey {name}! Ready to tackle some chemistry questions?",
            f"Hi {name}! Ask me anything about chemistry!"
        ]
        return random.choice(greetings)
    
    def generate_farewell(self):
        farewells = [
            "Goodbye! Feel free to come back anytime with more questions!",
            "See you later! Happy studying!",
            "Take care! Don't hesitate to ask if you need more help!",
            "Bye! Keep up the great work with your chemistry studies!"
        ]
        return random.choice(farewells)


class VASUChatbot:
    
    def __init__(self, ollama_model="mistral"):
        logger.info("Initializing V.A.H.N.I chatbot...")

        self.response_gen = ResponseGenerator(ollama_model)
        self.conversation = ConversationManager()
        self.intent_classifier = IntentClassifier()

        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TORCH_LOGS"] = ""
        
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_model.max_seq_length = 128

        self.knowledge_base = []
        self.embeddings = None
        self.cache_file = "vasu_embeddings_cache.pkl"
        self.topic_index = {}
        self.vocabulary = set()

        self.last_topic = None
        self.last_retrieved = None
        self.last_query = None
        self.user_name = "Krishnendu"

    def load_knowledge_base(self, json_path):
        logger.info(f"Loading knowledge base from {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.knowledge_base = data.get('training_data', [])
        logger.info(f"Loaded {len(self.knowledge_base)} knowledge base entries")

        for idx, item in enumerate(self.knowledge_base):
            if not isinstance(item, dict):
                continue
            
            topic = item.get('topic', '')
            query = item.get('user_query', '')
            
            if topic:
                topic_lower = topic.lower()
                self.topic_index.setdefault(topic_lower, []).append(idx)
                for word in topic_lower.split():
                    if len(word) > 2:
                        self.topic_index.setdefault(word, []).append(idx)
            
            if query:
                for word in query.lower().split():
                    if len(word) > 2:
                        self.topic_index.setdefault(word, []).append(idx)

            text = (query + ' ' + topic + ' ' + item.get('assistant_response', '')).lower()
            self.vocabulary.update(text.split())
        
        logger.info(f"Indexed {len(self.topic_index)} unique terms")

        if os.path.exists(self.cache_file):
            try:
                logger.info("Loading cached embeddings...")
                with open(self.cache_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                logger.info("Embeddings loaded from cache")
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
                self._create_embeddings()
        else:
            self._create_embeddings()

    def _create_embeddings(self):
        logger.info("Creating embeddings...")
        
        texts = []
        for item in self.knowledge_base:
            if not isinstance(item, dict):
                texts.append("")
                continue

            q = item.get('user_query', '')
            t = item.get('topic', '')
            a = item.get('assistant_response', '')[:300]
            combined = f"{q} {t} {t} {a}".strip()
            texts.append(combined if combined else "")
        
        self.embeddings = self.embedding_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        logger.info(f"Created embeddings with shape {self.embeddings.shape}")

        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            logger.info("Embeddings cached successfully")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")

    def spell_correct(self, query):
        corrected_words = []
        query_words = query.lower().split()
        corrections_made = False
        
        for word in query_words:
            if len(word) <= 2:
                corrected_words.append(word)
                continue
            
            if word in self.vocabulary:
                corrected_words.append(word)
                continue
            
            matches = get_close_matches(word, self.vocabulary, n=1, cutoff=0.85)
            if matches and matches[0] != word:
                corrected_words.append(matches[0])
                corrections_made = True
                logger.info(f"Corrected '{word}' to '{matches[0]}'")
            else:
                corrected_words.append(word)
        
        corrected_query = ' '.join(corrected_words)
        return corrected_query, corrections_made

    def find_topics(self, query, top_k=5):
        query_lower = query.lower()
        matches = []
        
        for topic, indices in self.topic_index.items():
            score = 0

            if query_lower == topic:
                score = 3
            elif query_lower in topic:
                score = 2
            elif any(word in topic for word in query_lower.split() if len(word) > 2):
                score = 1
            
            if score > 0:
                matches.append((topic, indices[0], score))

        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:top_k]

    def search(self, query, top_k=5):
        if self.embeddings is None or len(self.embeddings) == 0:
            raise RuntimeError("Embeddings not loaded.")

        corrected_query, corrected = self.spell_correct(query)
        if corrected:
            query = corrected_query
            logger.info(f"Using corrected query: {query}")

        q_emb = self.embedding_model.encode([query], convert_to_numpy=True)

        sims = cosine_similarity(q_emb, self.embeddings)[0]

        query_words = set(query.lower().split())
        for idx, item in enumerate(self.knowledge_base):
            topic_words = set(item.get('topic', '').lower().split())
            query_item_words = set(item.get('user_query', '').lower().split())

            exact_matches = len(query_words & (topic_words | query_item_words))
            if exact_matches > 0:
                sims[idx] += (exact_matches * 0.3)  

        top_k = min(top_k, len(sims))
        top_idx = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for idx in top_idx:
            try:
                item = self.knowledge_base[int(idx)]
                results.append((int(idx), float(sims[idx]), item))
            except Exception as e:
                logger.error(f"Error retrieving result {idx}: {e}")
                continue
        
        return results

    def is_balancing_query(self, query):
        q = query.lower()
        return ('balance' in q or 'balanced' in q) and ('->' in query or '→' in query)

    def build_prompt(self, query, retrieved, context_history=None):
        system = (
            "You are V.A.H.N.I, a chemistry tutor. "
            "Answer ONLY the current question asked. "
            "Do NOT repeat or re-explain any previous topics unless explicitly asked. "
            "Start your answer directly. No preamble like 'I'd be happy to...' or 'Sure!'. "
            "Be concise: 4-6 sentences max.\n\n"
            "For numerical problems:\n"
            "- Show clear step by step solution\n"
            "- Strong acids (HCl, HNO3, H2SO4) fully dissociate: [H+] = given concentration directly\n"
            "- Strong bases (NaOH, KOH) fully dissociate: [OH-] = given concentration directly\n"
            "- pH = -log[H+], pOH = -log[OH-], pH + pOH = 14\n"
            "- Never use Kw to find [H+] for strong acids, Kw is only for weak acids/bases\n"
            "- pH must always be between 0 and 14\n"
            "- State the final answer clearly at the end\n"
            "For balancing equations:\n"
            "- Write the unbalanced equation first\n"
            "- Count atoms as a simple list, no markdown tables\n"
            "- Show each balancing step on a new line\n"
            "- Simplify coefficients to lowest whole number ratio\n"
            "- Verify by recounting atoms at the end as a simple list\n"
            "- Never change chemical formulas, only coefficients\n"
            "- State the final balanced equation on its own line prefixed with 'Balanced:'\n"
            "If the user asks for an example or clarification, it must relate to the LAST topic discussed in context. "
            "Never switch to a different topic when asked for an example.\n\n"
        )

        history_block = ""
        if context_history:
            recent = context_history[-2:]
            lines = []
            for h in recent:
                q = h.get('query', '')
                topic = h.get('metadata', {}).get('topic', '')
                if q:
                    lines.append(f"- Previously asked about {topic}: {q}")
            if lines:
                history_block = "Conversation context (for follow-ups only):\n" + "\n".join(lines) + "\n\n"

        reference = ""
        if retrieved:
            top_item = retrieved[0][2]
            if isinstance(top_item, dict):
                text = top_item.get('assistant_response', '').strip()
                if len(text) > 800:
                    text = text[:800] + "..."
                if text:
                    reference = "For balancing equations, show ALL steps exactly as in the reference - atom counting, each balancing step, and final answer.\n"

        prompt = (
            system +
            history_block +
            reference +
            f"Current Question: {query}\n\n" +
            "Answer:"
        )
        return prompt
    
    def handle_greeting(self, intents):
        return self.response_gen.generate_greeting(self.user_name), {
            'intent': 'greeting',
            'time': '0.0s'
        }
    
    def handle_farewell(self, intents):
        return self.response_gen.generate_farewell(), {
            'intent': 'farewell',
            'time': '0.0s'
        }
    
    def handle_gratitude(self, intents):
        responses = [
            "You're welcome! Feel free to ask more questions anytime!",
            "Happy to help! Let me know if you need anything else!",
            "Glad I could help! What else would you like to know?",
            "No problem! I'm here whenever you need help with chemistry!"
        ]
        return random.choice(responses), {
            'intent': 'gratitude',
            'time': '0.0s'
        }
    
    def handle_confirmation(self, intents):
        return "Great! What else would you like to explore?", {
            'intent': 'confirmation',
            'time': '0.0s'
        }

    def chat(self, query):
        start = time.time()

        if not query or not query.strip():
            return "Please ask a question!", {'time': '0.0s'}
        
        query = query.strip()
        logger.info(f"Query: {query}")

        intents = self.intent_classifier.classify(query)
        logger.info(f"Detected intents: {intents}")

        if 'greeting' in intents:
            return self.handle_greeting(intents)
        
        if 'farewell' in intents:
            return self.handle_farewell(intents)

        # Follow-up detection
        q_lower = query.lower()
        is_followup = any(w in q_lower for w in [
            'example', 'give me', 'show me', 'demonstrate',
            'dont get', "don't get", 'didnt get', "didn't get",
            'clarify', 'elaborate', 'more detail', 'not understand',
            'confused', 'again', 'simpler', 'easier'
        ])

        if is_followup and self.last_topic and self.last_retrieved:
            prompt = (
                f"You are V.A.H.N.I, a chemistry tutor.\n"
                f"The student has been learning about: {self.last_topic}\n"
                f"They said: '{query}'\n"
                f"Give a clearer explanation or example ONLY about {self.last_topic}.\n"
                f"Do NOT bring up any other topic.\n\n"
                f"Reference: {self.last_retrieved[0][2].get('assistant_response', '')[:400]}\n\n"
                f"Response:"
            )
            response = self.response_gen.generate(prompt, temperature=0.4, max_tokens=400)
            meta = {
                'topic': self.last_topic,
                'time': f"{time.time()-start:.1f}s",
                'follow_up': True
            }
            self.conversation.add_interaction(query, response, meta)
            return response, meta

        if self.is_balancing_query(query):
            equation = re.sub(r'(?i)balance\s*(this\s*)?(equation\s*)?:?\s*', '', query).strip()
            result = format_balanced_equation(equation)
            if 'error' not in result:
                return result['steps'], {'topic': 'Balancing Equations', 'time': f"{time.time()-start:.1f}s"}

        if len(query.split()) <= 3:
            if 'gratitude' in intents:
                return self.handle_gratitude(intents)
            if 'confirmation' in intents:
                return self.handle_confirmation(intents)

        if len(query.split()) <= 2 and 'definition' not in intents:
            topics = self.find_topics(query)
            if topics:
                response = f"I found these topics related to '{query}':\n\n"
                for i, (topic, idx, score) in enumerate(topics, 1):
                    chapter = ""
                    if 0 <= idx < len(self.knowledge_base):
                        chapter = self.knowledge_base[idx].get('chapter', '')
                    response += f"{i}. {topic}"
                    if chapter:
                        response += f" ({chapter})"
                    response += "\n"
                response += "\nWhich topic would you like to learn about?"
                meta = {
                    'needs_clarification': True,
                    'topics_found': len(topics),
                    'time': f"{time.time() - start:.1f}s"
                }
                return response, meta

        try:
            retrieved = self.search(query, top_k=5)
            logger.info(f"Retrieved {len(retrieved)} results")
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Search error: {e}", {'time': f"{time.time()-start:.1f}s"}
        
        if not retrieved:
            return "I couldn't find relevant information. Could you rephrase your question?", {
                'time': f"{time.time()-start:.1f}s"
            }

        first = retrieved[0]
        context_item = first[2] if len(first) >= 3 else {}
        top_score = float(first[1]) if len(first) >= 2 else 0.0

        self.last_topic = context_item.get('topic', '') if isinstance(context_item, dict) else ''
        self.last_retrieved = retrieved
        self.last_query = query

        context_history = self.conversation.get_context(3)
        prompt = self.build_prompt(query, retrieved, context_history)

        temp = 0.5 if 'example' in intents else 0.3
        response = self.response_gen.generate(prompt, temperature=temp, max_tokens=600)

        meta = {
            'topic': context_item.get('topic', '') if isinstance(context_item, dict) else '',
            'chapter': context_item.get('chapter', '') if isinstance(context_item, dict) else '',
            'score': f"{top_score:.3f}",
            'time': f"{time.time() - start:.1f}s",
            'intents': intents,
            'results_count': len(retrieved)
        }

        self.conversation.add_interaction(query, response, meta)
        
        logger.info(f"Response generated in {time.time()-start:.1f}s")
        return response, meta
    
    def clear_history(self):
        self.conversation.clear()
        self.last_topic = None
        self.last_retrieved = None
        self.last_query = None
        logger.info("Conversation history cleared")
    
    def get_stats(self):
        """Get chatbot statistics"""
        return self.conversation.get_stats()

chatbot = None


class RequestHandler(BaseHTTPRequestHandler):
    
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _send_html(self, content, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(content.encode())
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path.lstrip('/')

        file_path = path if path else 'index.html'
        
        if os.path.exists(file_path) and os.path.isfile(file_path):
            if file_path.endswith('.html'):
                ctype = 'text/html'
            elif file_path.endswith('.css'):
                ctype = 'text/css'
            elif file_path.endswith('.js'):
                ctype = 'application/javascript'
            elif file_path.endswith('.png'):
                ctype = 'image/png'
            elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
                ctype = 'image/jpeg'
            else:
                ctype = 'application/octet-stream'
            
            self.send_response(200)
            self.send_header('Content-type', ctype)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
            return
        

        if path.startswith('chat'):
            query_string = parsed_path.query
            params = parse_qs(query_string)
            query = params.get('query', [''])[0]
            
            if not query:
                self._send_json({
                    'status': 'error',
                    'response': 'No query provided',
                    'meta': {}
                }, 400)
                return
            
            try:
                response, meta = chatbot.chat(query)
                self._send_json({
                    'status': 'success',
                    'response': response,
                    'meta': meta
                })
            except Exception as e:
                logger.error(f"Chat error: {e}", exc_info=True)
                self._send_json({
                    'status': 'error',
                    'response': f'Error: {str(e)}',
                    'meta': {}
                }, 500)
            return
        
        elif path == 'clear':
            try:
                chatbot.clear_history()
                self._send_json({
                    'status': 'success',
                    'message': 'Chat history cleared'
                })
            except Exception as e:
                logger.error(f"Clear error: {e}")
                self._send_json({
                    'status': 'error',
                    'message': str(e)
                }, 500)
            return
        
        elif path == 'stats':
            try:
                stats = chatbot.get_stats()
                self._send_json({
                    'status': 'success',
                    'stats': stats
                })
            except Exception as e:
                logger.error(f"Stats error: {e}")
                self._send_json({
                    'status': 'error',
                    'message': str(e)
                }, 500)
            return
        
        elif path == 'health':
            self._send_json({
                'status': 'healthy',
                'service': 'V.A.H.N.I',
                'timestamp': datetime.now().isoformat()
            })
            return
        
        self._send_json({
            'status': 'error',
            'message': 'Endpoint not found'
        }, 404)
    
    def log_message(self, format, *args):
        """Custom logging"""
        logger.info(f"{self.address_string()} - {format % args}")


def run_server(port=8000):
    server = HTTPServer(('0.0.0.0', port), RequestHandler)
    
    localhost_url = f"http://localhost:{port}"
    
    try:
        hostname = socket.gethostbyname(socket.gethostname())
        lan_url = f"http://{hostname}:{port}"
    except Exception:
        lan_url = None
    
    logger.info(f"V.A.H.N.I server running at {localhost_url}")
    if lan_url:
        logger.info(f"Access from local network: {lan_url}")
    logger.info("Press Ctrl+C to stop the server")

    try:
        webbrowser.open(localhost_url)
        logger.info("Opening default browser...")
    except Exception as e:
        logger.warning(f"Could not open browser automatically: {e}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
        server.shutdown()


if __name__ == "__main__":
    import sys
    
    dataset = sys.argv[1] if len(sys.argv) > 1 else "chemistry_dataset.json"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    if not os.path.exists(dataset):
        logger.error(f"Dataset not found: {dataset}")
        sys.exit(1)

    logger.info("V.A.H.N.I - Virtual Assistant Helping Novel Insights")
    
    chatbot = VASUChatbot()
    chatbot.load_knowledge_base(dataset)
    
    logger.info("Chatbot initialized successfully!")
    logger.info(f"Loaded {len(chatbot.knowledge_base)} knowledge entries")
    logger.info(f"Indexed {len(chatbot.topic_index)} terms")
    
    run_server(port)