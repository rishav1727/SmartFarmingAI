import os
import json
import numpy as np
from openai import OpenAI
from src.consts import MODEL_DIR

class SmartAdvisor:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            print("WARNING: OPENAI_API_KEY not found in environment. Running in Offline Mock mode.")
            
        self.doc_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "documents")
        os.makedirs(self.doc_dir, exist_ok=True)
        
        self.index_path_embeddings = os.path.join(MODEL_DIR, "rag_embeddings.npy")
        self.index_path_chunks = os.path.join(MODEL_DIR, "rag_chunks.json")
        
        self.chunks = []
        self.embeddings = None
        self.load_index()

    def load_index(self):
        """Load RAG index from cache if it exists."""
        if os.path.exists(self.index_path_embeddings) and os.path.exists(self.index_path_chunks):
            try:
                self.embeddings = np.load(self.index_path_embeddings)
                with open(self.index_path_chunks, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                print(f"Loaded RAG index with {len(self.chunks)} chunks.")
            except Exception as e:
                print(f"Error loading RAG index: {e}")

    def index_documents(self):
        """Index all text and markdown files in the documents directory."""
        if not self.client:
            print("Cannot index documents: OpenAI client not initialized.")
            return False

        all_chunks = []
        
        # Load and chunk files
        for filename in os.listdir(self.doc_dir):
            file_path = os.path.join(self.doc_dir, filename)
            if os.path.isfile(file_path) and (filename.endswith(".txt") or filename.endswith(".md")):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    # Basic chunking: split by paragraphs or double newlines
                    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                    for p_idx, p in enumerate(paragraphs):
                        # Chunk size constraint
                        if len(p) > 1000:
                            # Split large paragraphs into smaller sentences
                            sentences = p.split(". ")
                            temp_chunk = ""
                            for s in sentences:
                                if len(temp_chunk) + len(s) < 800:
                                    temp_chunk += s + ". "
                                else:
                                    all_chunks.append({
                                        "source": filename,
                                        "chunk_id": f"{filename}_{p_idx}_{len(all_chunks)}",
                                        "text": temp_chunk.strip()
                                    })
                                    temp_chunk = s + ". "
                            if temp_chunk:
                                all_chunks.append({
                                    "source": filename,
                                    "chunk_id": f"{filename}_{p_idx}_{len(all_chunks)}",
                                    "text": temp_chunk.strip()
                                })
                        else:
                            all_chunks.append({
                                "source": filename,
                                "chunk_id": f"{filename}_{p_idx}",
                                "text": p
                            })
                except Exception as e:
                    print(f"Error reading file {filename} for indexing: {e}")
            elif os.path.isfile(file_path) and filename.endswith(".pdf"):
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(file_path)
                    for page_idx, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            # Chunk by paragraphs inside page text
                            paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
                            for p_idx, p in enumerate(paragraphs):
                                if len(p) > 1000:
                                    sentences = p.split(". ")
                                    temp_chunk = ""
                                    for s in sentences:
                                        if len(temp_chunk) + len(s) < 800:
                                            temp_chunk += s + ". "
                                        else:
                                            all_chunks.append({
                                                "source": filename,
                                                "chunk_id": f"{filename}_page_{page_idx}_p_{p_idx}_{len(all_chunks)}",
                                                "text": temp_chunk.strip()
                                            })
                                            temp_chunk = s + ". "
                                    if temp_chunk:
                                        all_chunks.append({
                                            "source": filename,
                                            "chunk_id": f"{filename}_page_{page_idx}_p_{p_idx}_{len(all_chunks)}",
                                            "text": temp_chunk.strip()
                                        })
                                else:
                                    all_chunks.append({
                                        "source": filename,
                                        "chunk_id": f"{filename}_page_{page_idx}_p_{p_idx}",
                                        "text": p
                                    })
                except Exception as e:
                    print(f"Error reading PDF file {filename} for indexing: {e}")
                    
        if not all_chunks:
            print("No document chunks found to index.")
            return False

        # Generate embeddings
        print(f"Generating embeddings for {len(all_chunks)} document chunks...")
        try:
            texts = [chunk["text"] for chunk in all_chunks]
            
            # OpenAI embedding API
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            
            embeddings_list = [data.embedding for data in response.data]
            self.embeddings = np.array(embeddings_list, dtype=np.float32)
            self.chunks = all_chunks
            
            # Save cache
            os.makedirs(MODEL_DIR, exist_ok=True)
            np.save(self.index_path_embeddings, self.embeddings)
            with open(self.index_path_chunks, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, indent=4)
                
            print(f"Successfully cached RAG index with {len(self.chunks)} chunks.")
            return True
        except Exception as e:
            print(f"Error generating RAG embeddings: {e}")
            return False

    def retrieve_context(self, query, top_k=3):
        """Retrieve relevant context chunks based on query embedding."""
        if self.embeddings is None or not self.chunks or not self.client:
            return []
            
        try:
            # Embed query
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=[query]
            )
            query_emb = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Cosine similarity (since embeddings are normalized, dot product is cosine similarity)
            similarities = np.dot(self.embeddings, query_emb)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            retrieved = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score > 0.3:  # Only return reasonably matched chunks
                    retrieved.append({
                        "chunk": self.chunks[idx],
                        "score": score
                    })
            return retrieved
        except Exception as e:
            print(f"Error in RAG retrieval: {e}")
            return []

    def get_treatment_advice(self, disease_name, confidence, language="English"):
        """Get structured treatment recommendations, leveraging RAG if available."""
        # Standardize disease name by replacing underscores
        clean_disease_name = disease_name.replace("___", " - ").replace("_", " ")
        
        # Check for OOD / Safety Fallbacks
        if "not a specialized plant" in clean_disease_name.lower() or "uncertain" in clean_disease_name.lower() or "unknown" in clean_disease_name.lower():
            is_hindi = language.lower() == "hindi"
            if is_hindi:
                return {
                    "overview": f"असंगति पाई गई: छवि पौधे की पत्ती जैसी नहीं है ({clean_disease_name})।",
                    "chemical": "अनुपलब्ध: गैर-कृषि वस्तुओं के लिए कोई रासायनिक नियंत्रण संभव नहीं है।",
                    "biological": "अनुपलब्ध: जैविक उपचार केवल फसलों और पौधों पर लागू होते हैं।",
                    "preventative": "सलाह: कृपया पौधे की पत्ती की एक स्पष्ट, निकट से खींची गई तस्वीर अपलोड करें।",
                    "advice": "निदान चलाने के लिए केवल कृषि पौधों की पत्तियों की छवियां अपलोड करें।"
                }
            else:
                return {
                    "overview": f"Anomaly Detected: The uploaded image does not resemble a valid plant leaf ({clean_disease_name}).",
                    "chemical": "N/A: No chemical treatment can be recommended for non-agricultural objects.",
                    "biological": "N/A: Organic and biological controls only apply to botanical crops.",
                    "preventative": "Guidance: Please upload a clear, high-resolution close-up photo of a crop leaf.",
                    "advice": "To run a valid diagnosis, please submit leaf images for supported agricultural plants."
                }
        
        # 1. RAG search
        context_str = ""
        retrieved_items = self.retrieve_context(clean_disease_name, top_k=2)
        if retrieved_items:
            context_chunks = [f"[Source: {item['chunk']['source']}] {item['chunk']['text']}" for item in retrieved_items]
            context_str = "\n\n".join(context_chunks)
            print(f"RAG context retrieved for '{clean_disease_name}': {len(retrieved_items)} chunks.")
            
        # 2. Call OpenAI or fall back to mock
        if not self.client:
            return self._get_mock_advice(clean_disease_name, confidence, language)
            
        try:
            system_prompt = (
                "You are an expert AI Agronomist and Plant Pathologist assisting farmers. "
                "Provide detailed, scientific, and highly practical treatment recommendations for the diagnosed crop disease. "
                "You must respond with a JSON object containing the following keys (DO NOT wrap in Markdown formatting except the JSON block if required, but return raw JSON or a JSON codeblock):\n"
                "{\n"
                "  \"overview\": \"Description of the disease, symptoms, causes, and crop impact.\",\n"
                "  \"chemical\": \"Chemical treatments (fungicides/pesticides/dosages).\",\n"
                "  \"biological\": \"Organic/biological treatments (neem oil, biocontrol agents, homemade sprays).\",\n"
                "  \"preventative\": \"Cultural practices, preventative checks, and soil management tips.\",\n"
                "  \"advice\": \"General recommendation and immediate next steps for the farmer.\"\n"
                "}\n"
                f"Generate the response content in the selected language: {language}. "
                "Ensure language translation is highly natural, professional, and easily understandable by a local farmer."
            )
            
            user_prompt = f"Diagnosed Crop & Disease: {clean_disease_name}\nConfidence Score: {confidence:.2f}%\n"
            if context_str:
                user_prompt += f"\nRelevant Agricultural Database Reference Context:\n{context_str}\n\nUse this context where applicable to enrich the advice."
                
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            advice_json = json.loads(response.choices[0].message.content)
            return advice_json
            
        except Exception as e:
            print(f"OpenAI API error: {e}. Falling back to Mock.")
            return self._get_mock_advice(clean_disease_name, confidence, language)

    def chat_about_disease(self, disease_name, chat_history, user_message, language="English"):
        """Conversational Q&A about the diagnosed crop disease."""
        clean_disease_name = disease_name.replace("___", " - ").replace("_", " ")
        
        if not self.client:
            return f"Offline Mode: I see you are asking about '{clean_disease_name}'. Please setup the OpenAI API key to enable interactive chat advice."
            
        try:
            # Retrieve RAG context if relevant
            context_str = ""
            retrieved_items = self.retrieve_context(user_message, top_k=2)
            if retrieved_items:
                context_chunks = [f"[Source: {item['chunk']['source']}] {item['chunk']['text']}" for item in retrieved_items]
                context_str = "\n\n".join(context_chunks)

            system_prompt = (
                f"You are a helpful AI Agronomist chatbot. The farmer's crop was diagnosed with '{clean_disease_name}'. "
                "Answer the farmer's question friendly and practically. Keep explanations simple and action-oriented. "
                f"Respond in the language: {language}. If the language is Hindi, write in standard Devanagari script."
            )
            if context_str:
                system_prompt += (
                    f"\nUse this local document reference context to answer the question: \n{context_str}\n"
                    "At the end of your response, always cite the source file names you used, formatted exactly as:\n"
                    "For English: '(Source: filename)'\n"
                    "For Hindi: '(स्रोत: filename)'"
                )
                
            messages = [{"role": "system", "content": system_prompt}]
            
            # Format chat history
            for msg in chat_history[-6:]:  # Limit context window to last 6 messages
                role = "user" if msg["sender"] == "user" else "assistant"
                messages.append({"role": role, "content": msg["text"]})
                
            messages.append({"role": "user", "content": user_message})
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.5
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Chat completion error: {e}")
            return "I apologize, I am having trouble connecting to my knowledge base right now. Please try again in a moment."

    def _get_mock_advice(self, disease_name, confidence, language):
        """Fallback advice for offline mode."""
        is_hindi = language.lower() == "hindi"
        
        if is_hindi:
            return {
                "overview": f"{disease_name} का संक्रमण पाया गया है। (विश्वास: {confidence:.1f}%)। यह रोग पत्तियों पर धब्बे और पीलापन लाता है, जिससे प्रकाश संश्लेषण प्रभावित होता है।",
                "chemical": "1. कॉपर ऑक्सीक्लोराइड 50% WP @ 2.5 ग्राम प्रति लीटर पानी में मिलाकर स्प्रे करें।\n2. यदि रोग गंभीर है, तो कवकनाशी (Fungicide) का छिड़काव करें।",
                "biological": "1. नीम तेल (Neem Oil) 5 मिली प्रति लीटर पानी में लिक्विड सोप के साथ मिलाकर 7-10 दिनों के अंतराल पर छिड़कें।\n2. संक्रमित पत्तियों को हटाकर नष्ट कर दें।",
                "preventative": "1. खेत में उचित जल निकासी (Water drainage) सुनिश्चित करें।\n2. पौधों के बीच हवा का संचार बढ़ाने के लिए दूरी बनाए रखें।\n3. नाइट्रोजन उर्वरकों के अत्यधिक उपयोग से बचें।",
                "advice": "प्रभावित पौधों के हिस्सों को तुरंत हटा दें और कवकनाशी का छिड़काव सुबह या शाम के समय करें।"
            }
        else:
            return {
                "overview": f"Diagnosed with {disease_name} (Confidence: {confidence:.1f}%). This condition affects the foliage, leading to lesions, discoloration, and reduced photosynthesis capacity.",
                "chemical": "1. Apply Copper Oxychloride 50% WP at 2.5g per liter of water.\n2. In case of severe infection, spray a systemic fungicide (e.g., Mancozeb or Tebuconazole) as per instructions.",
                "biological": "1. Spray Neem Oil (5ml/L of water) mixed with a few drops of mild soap at 7-10 day intervals.\n2. Prune and destroy heavily infected leaves to halt spore dispersal.",
                "preventative": "1. Ensure proper crop spacing to enhance air circulation.\n2. Avoid overhead watering; use drip irrigation to keep foliage dry.\n3. Balance soil nutrients, avoiding excessive nitrogen.",
                "advice": "Immediately isolate/prune infected parts. Apply preventive neem oil spray and monitor watering closely."
            }
