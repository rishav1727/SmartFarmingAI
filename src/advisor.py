import os
import json
import time
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
        self.index_path_chunks     = os.path.join(MODEL_DIR, "rag_chunks.json")
        self.cache_path            = os.path.join(MODEL_DIR, "advice_cache.json")

        # ── Pre-computed advice cache (PRIMARY — <2ms lookup) ─────────────
        self._advice_cache = {}
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._advice_cache = json.load(f)
                print(f"Loaded advice cache: {len(self._advice_cache)} diseases (instant lookup enabled)")
            except Exception as e:
                print(f"Warning: Could not load advice cache ({e}). Falling back to LLM.")

        self.chunks    = []
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

        # Save chunks
        self.chunks = all_chunks

        # Generate embeddings if OpenAI client is available
        print(f"Generating embeddings for {len(all_chunks)} document chunks...")
        try:
            if not self.client:
                raise ValueError("OpenAI client not configured.")
                
            texts = [chunk["text"] for chunk in all_chunks]
            
            # OpenAI embedding API
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            
            embeddings_list = [data.embedding for data in response.data]
            self.embeddings = np.array(embeddings_list, dtype=np.float32)
            
            # Save cache
            os.makedirs(MODEL_DIR, exist_ok=True)
            np.save(self.index_path_embeddings, self.embeddings)
            with open(self.index_path_chunks, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, indent=4)
                
            print(f"Successfully cached vector RAG index with {len(self.chunks)} chunks.")
            return True
        except Exception as e:
            print(f"Notice: RAG Vector embeddings fallback active ({e}). Defaulting to Keyword RAG Index.")
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(self.index_path_chunks, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, indent=4)
            return True

    def retrieve_context(self, query, top_k=3):
        """Retrieve relevant context chunks based on query embedding or keyword fallback."""
        if not self.chunks:
            return []
            
        if self.embeddings is not None and self.client:
            try:
                # Embed query
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[query]
                )
                query_emb = np.array(response.data[0].embedding, dtype=np.float32)
                
                # Cosine similarity
                similarities = np.dot(self.embeddings, query_emb)
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                retrieved = []
                for idx in top_indices:
                    score = float(similarities[idx])
                    if score > 0.3:
                        retrieved.append({
                            "chunk": self.chunks[idx],
                            "score": score
                        })
                return retrieved
            except Exception as e:
                print(f"Vector search fallback active ({e}). Using Keyword RAG Search...")

        # Fallback Keyword / Token Search
        query_terms = [term for term in query.lower().split() if len(term) > 3]
        retrieved = []
        for chunk in self.chunks:
            text_lower = chunk["text"].lower()
            score = sum(1 for term in query_terms if term in text_lower)
            if score > 0:
                retrieved.append({
                    "chunk": chunk,
                    "score": float(score)
                })
        retrieved.sort(key=lambda x: x["score"], reverse=True)
        return retrieved[:top_k]

    def get_treatment_advice(self, disease_name, confidence, language="English"):
        """Get structured treatment recommendations — Cache → RAG+LLM → Mock fallback."""
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

        # ── 1. PRE-COMPUTED CACHE LOOKUP (<2ms) ───────────────────────────
        # Try exact disease_name key first, then clean_disease_name
        _t0 = time.perf_counter()
        for _key in [disease_name, clean_disease_name]:
            if _key in self._advice_cache:
                lang_data = self._advice_cache[_key]
                # Language match: exact → English fallback
                result = lang_data.get(language) or lang_data.get("English")
                if result:
                    _ms = (time.perf_counter() - _t0) * 1000
                    print(f"Cache HIT: '{_key}' [{language}] in {_ms:.2f}ms")
                    return result

        print(f"Cache MISS for '{disease_name}' [{language}] — falling back to LLM/Mock")

        # ── 2. RAG search ─────────────────────────────────────────────────
        context_str = ""
        retrieved_items = self.retrieve_context(clean_disease_name, top_k=2)
        if retrieved_items:
            context_chunks = [f"[Source: {item['chunk']['source']}] {item['chunk']['text']}" for item in retrieved_items]
            context_str = "\n\n".join(context_chunks)
            print(f"RAG context retrieved for '{clean_disease_name}': {len(retrieved_items)} chunks.")

        # ── 3. Call OpenAI or fall back to mock ───────────────────────────
        if not self.client:
            return self._get_mock_advice(clean_disease_name, confidence, language)

            
        try:
            system_prompt = (
                "You are an expert AI Agronomist and Plant Pathologist assisting farmers. "
                "Provide highly elaborate, comprehensive, and practical treatment recommendations for the diagnosed crop disease. "
                "CRITICAL: Use HTML bold tags <b>...</b> or Markdown bold **...** around all key treatments, chemical names, precise spray dosages, pathogen names, and critical preventative steps so that the farmer can immediately identify them. "
                "You must respond with a JSON object containing the following keys (DO NOT wrap in Markdown formatting except the JSON block if required):\n"
                "{\n"
                "  \"overview\": \"Elaborate description of the disease, pathogen causes, leaf spot symptoms, environmental triggers, and potential crop yield loss impact.\",\n"
                "  \"chemical\": \"Comprehensive chemical treatments (exact fungicide/pesticide names, dosages per liter/acre, and application intervals highlighted in BOLD).\",\n"
                "  \"biological\": \"Detailed organic and biological controls (neem oil concentrations, bio-fungicides, natural sprays, pruning instructions highlighted in BOLD).\",\n"
                "  \"preventative\": \"Cultural practices, field hygiene, drip irrigation guidance, and soil nutrient management tips highlighted in BOLD.\",\n"
                "  \"advice\": \"Immediate key recommendation summary for the farmer.\"\n"
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
                "Answer the farmer's question friendly and practically. Keep explanations simple and action-oriented. Use bold text for key dosages and product names. "
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
        """Fallback advice for offline mode supporting all major Indian local languages."""
        lang_lower = language.lower() if language else "english"
        
        if lang_lower in ["hindi", "marathi", "maithili", "nepali", "kashmiri", "dogri", "bodo", "sanskrit"]:
            return {
                "overview": f"<b>रोग निदान:</b> <b>{disease_name}</b> का संक्रमण पुष्टि हुआ (विश्वास: <b>{confidence:.1f}%</b>)।\n\n"
                            f"<b>लक्षण एवं प्रभाव:</b> यह फंगल/बैक्टीरियल संक्रमण पत्तियों पर काले-भूरे धब्बे, किनारे सूखने और पत्तियों के समय से पहले झड़ने का कारण बनता है। "
                            f"यदि आर्द्र मौसम में उपचार न किया जाए, तो यह फसल उपज में <b>30% से 50% तक नुकसान</b> कर सकता है।",
                "chemical": "<b>संस्तुत रासायनिक नियंत्रण उपाय:</b>\n"
                            "1. <b>कॉपर ऑक्सीक्लोराइड 50% WP</b> @ <b>2.5 ग्राम प्रति लीटर पानी</b> (500 ग्राम/एकड़) का छिड़काव करें।\n"
                            "2. <b>गंभीर संक्रमण में:</b> <b>मैन्कोज़ेब 75% WP</b> (2 ग्राम/लीटर) या <b>टेबूकोनाज़ोल</b> (1 मिली/लीटर) का प्रयोग करें।\n"
                            "3. <b>छिड़काव समय:</b> सुबह या शाम के समय छिड़काव करें। <b>10-12 दिनों बाद</b> आवश्यकतानुसार दोहराएं।",
                "biological": "<b>जैविक व जैविक नियंत्रण उपाय:</b>\n"
                            "1. <b>नीम तेल छिड़काव:</b> <b>नीम तेल (10,000 PPM) @ 5 मिली/लीटर पानी</b> साबुन के घोल के साथ 7-10 दिनों के अंतराल पर छिड़कें।\n"
                            "2. <b>बायो-फंगीसाइड:</b> <b>ट्राइकोडरमा विरिडे</b> या <b>स्यूडोमोनास फ्लोरेसेंस</b> (5 ग्राम/लीटर) का पर्णीय छिड़काव करें।\n"
                            "3. <b>संक्रमित पत्तियां:</b> अत्यधिक प्रभावित पत्तियों को तोड़कर नष्ट कर दें।",
                "preventative": "<b>रोकथाम व कृषि सावधानियां:</b>\n"
                            "1. <b>उचित जल निकासी:</b> खेत में पानी रुकने न दें और <b>ड्रिप सिंचाई</b> का प्रयोग करें।\n"
                            "2. <b>पौधों की दूरी:</b> धूप और हवा का आवागमन बढ़ाने के लिए पौधों में उचित दूरी रखें।\n"
                            "3. <b>संतुलित पोषण:</b> अत्यधिक नाइट्रोजन उर्वरक से बचें; <b>पोटेशियम और जैविक खाद</b> का उपयोग करें।",
                "advice": "प्रभावित भागों को तुरंत हटाएं, <b>कॉपर या नीम का घोल</b> छिड़कें और सिंचाई पर ध्यान दें।"
            }
        elif lang_lower in ["bengali", "assamese", "manipuri", "santali"]:
            return {
                "overview": f"<b>রোগ শনাক্তকরণ:</b> <b>{disease_name}</b> শনাক্ত করা হয়েছে (আত্মবিশ্বাস: <b>{confidence:.1f}%</b>)।\n\n"
                            f"<b>লক্ষণ ও প্রভাব:</b> এই ছত্রাক/ব্যাকটেরিয়া জনিত রোগ পাতায় কালো-বাদামী দাগ এবং পাতা ঝরে পড়ার কারণ হয়। "
                            f"সময়মতো চিকিৎসা না করলে ফসলের ফলন <b>৩০% থেকে ৫০% হ্রাস</b> পেতে পারে।",
                "chemical": "<b>রাসায়নিক নিয়ন্ত্রণ ব্যবস্থা:</b>\n"
                            "১. <b>কপার অক্সিক্লোরাইড ৫০% WP</b> প্রতি লিটার পানিতে <b>২.৫ গ্রাম</b> মিশিয়ে স্প্রে করুন।\n"
                            "২. <b>তীব্র সংক্রমণে:</b> <b>ম্যানকোজেব ৭৫% WP</b> (২ গ্রাম/লিটার) প্রয়োগ করুন।\n"
                            "৩. <b>সময়সূচী:</b> সকালে বা বিকেলে স্প্রে করুন এবং <b>১০-১২ দিন পর</b> পুনরাবৃত্তি করুন।",
                "biological": "<b>জৈব ও প্রাকৃতিক প্রতিকার:</b>\n"
                            "১. <b>নিম তেল স্প্রে:</b> <b>নিম তেল (৫ মিলি/লিটার)</b> সাবান পানির সাথে মিশিয়ে স্প্রে করুন।\n"
                            "২. <b>বায়ো-ফাঙ্গিসাইড:</b> <b>ট্রাইকোডার্মা ভিরিডি</b> (৫ গ্রাম/লিটার) প্রয়োগ করুন।",
                "preventative": "<b>প্রতিরোধমূলক ব্যবস্থা:</b>\n"
                            "১. ক্ষেতে জল নিষ্কাশন ব্যবস্থা উন্নত করুন এবং <b>ড্রিপ সেচ</b> ব্যবহার করুন।\n"
                            "২. পর্যাপ্ত আলো বাতাসের জন্য গাছের মাঝে সঠিক দূরত্ব রাখুন।",
                "advice": "আক্রান্ত পাতা অপসারণ করুন এবং অবিলম্বে <b>ছত্রাকনাশক স্প্রে</b> করুন।"
            }
        elif lang_lower in ["telugu", "kannada", "tamil", "malayalam", "konkani"]:
            return {
                "overview": f"<b>వ్యాధి నిర్ధారణ:</b> <b>{disease_name}</b> గుర్తించబడింది (విశ్వసనీయత: <b>{confidence:.1f}%</b>).\n\n"
                            f"<b>లక్షణాలు & ప్రభావం:</b> ఈ శిలీంధ్ర/బ్యాక్టీరియా సంక్రామకం ఆకులపై నల్లటి మచ్చలు మరియు ఆకులు రాలిపోవడానికి కారణమవుతుంది. "
                            f"నివారణ చర్యలు తీసుకోకపోతే దిగుబడి <b>30% నుండి 50% వరకు</b> తగ్గే ప్రమాదం ఉంది.",
                "chemical": "<b>రసాయనిక నివారణ చర్యలు:</b>\n"
                            "1. <b>కాపర్ ఆక్సిక్లోరైడ్ 50% WP</b> లీటరు నీటికి <b>2.5 గ్రాములు</b> కలిపి పిచికారీ చేయండి.\n"
                            "2. <b>తీవ్రత ఎక్కువగా ఉంటే:</b> <b>మ్యాంకోజెబ్ 75% WP</b> (2 గ్రా/లీ) ఉపయోగించండి.\n"
                            "3. <b>పిచికారీ సమయం:</b> ఉదయం లేదా సాయంత్రం వేళల్లో పిచికారీ చేసి <b>10-12 రోజులకు</b> పునరావృతం చేయండి.",
                "biological": "<b>జైవిక & ఆర్గానిక్ నివారణ:</b>\n"
                            "1. <b>వేప నూనె స్ప్రే:</b> <b>వేప నూనె (5 మి.లీ/లీటరు)</b> సబ్బు నీటితో కలిపి పిచికారీ చేయండి.\n"
                            "2. <b>ట్రైకోడెర్మా విరిడే</b> (5 గ్రా/లీ) పిచికారీ చేయండి.",
                "preventative": "<b>ముందస్తు జాగ్రత్తలు:</b>\n"
                            "1. పొలంలో నీరు నిల్వ ఉండకుండా చూసుకోండి మరియు <b>బిందు సేద్యం (Drip)</b> వాడండి.\n"
                            "2. మొక్కల మధ్య సరైన దూరం పాటించండి.",
                "advice": "వ్యాధి సోకిన ఆకులను తొలగించి, వెంటనే <b>ఫంగిసైడ్ పిచికారీ</b> చేయండి."
            }
        elif lang_lower in ["gujarati", "punjabi", "odia", "urdu"]:
            return {
                "overview": f"<b>રોગ નિદાન:</b> <b>{disease_name}</b> જણાયેલ છે (વિશ્વાસ: <b>{confidence:.1f}%</b>).\n\n"
                            f"<b>લક્ષણો અને અસર:</b> આ રોગ પાંદડા પર ડાઘ, પીળાશ અને અકાળે પાંદડા ખરી પડવાનું કારણ બને છે. "
                            f"સમયસર સારવાર ન કરવાથી <b>30% થી 50% ઉત્પાદન ઘટી</b> શકે છે.",
                "chemical": "<b>રાસાયણિક નિયંત્રણ પગલાં:</b>\n"
                            "1. <b>કોપર ઓક્સીક્લોરાઇડ 50% WP</b> પ્રતિ લીટર પાણીમાં <b>2.5 ગ્રામ</b> ભેળવીને છંટકાવ કરો.\n"
                            "2. <b>વધારે સંક્રમણમાં:</b> <b>મેન્કોઝેબ 75% WP</b> (2 ગ્રામ/લીટર) વાપરો.\n"
                            "3. <b>છંટકાવ સમય:</b> સવારે અથવા સાંજે છંટકાવ કરો અને <b>10-12 દિવસ પછી</b> પુનરાવર્તન કરો.",
                "biological": "<b>જૈવિક સારવાર:</b>\n"
                            "1. <b>લીમડાનું તેલ:</b> <b>લીમડાનું તેલ (5 મિલી/લીટર)</b> સાબુના પાણી સાથે છાંટો.\n"
                            "2. સંક્રમિત પાંદડા કાપીને નાશ કરો.",
                "preventative": "<b>સાવચેતીના પગલાં:</b>\n"
                            "1. ખેતરમાં પાણીનો યોગ્ય નિકાલ કરો અને <b>ડ્રિપ પદ્ધતિ</b> અપનાવો.\n"
                            "2. છોડ વચ્ચે યોગ્ય અંતર રાખો.",
                "advice": "રોગગ્રસ્ત ભાગો દૂર કરો અને <b>ફુગનાશકનો છંટકાવ</b> કરો."
            }
        else:
            return {
                "overview": f"<b>Disease Identification:</b> Diagnosed with <b>{disease_name}</b> (Confidence Score: <b>{confidence:.1f}%</b>).\n\n"
                            f"<b>Pathology & Impact:</b> This phytopathogenic infection affects foliage, producing necrotic spots, chlorotic yellow halos, and premature leaf drop. "
                            f"Under high humidity, unmanaged infection can lead to <b>30% to 60% potential yield reduction</b>.",
                "chemical": "<b>Recommended Chemical Control Plan:</b>\n"
                            "• <b>Foliar Fungicide Spray:</b> Apply <b>Copper Oxychloride 50% WP</b> at <b>2.5g per Liter of water</b> (approx. 500g per acre).\n"
                            "• <b>Systemic Fungicide (Severe Spread):</b> Rotate with <b>Mancozeb 75% WP</b> (2g/L) or <b>Tebuconazole 25.9% EC</b> (1ml/L).\n"
                            "• <b>Application Window:</b> Spray early morning or late evening. Re-apply after <b>10 to 12 days</b> if heavy rain occurs.",
                "biological": "<b>Organic & Biological Remedies:</b>\n"
                            "• <b>Neem Oil Spray:</b> Mix <b>Neem Oil (10,000 PPM) at 5ml/Liter</b> with 1ml liquid emulsifier.\n"
                            "• <b>Biocontrol Agent:</b> Apply foliar spray of <b>Pseudomonas fluorescens</b> or <b>Trichoderma viride</b> (5g/L).\n"
                            "• <b>Sanitation:</b> Immediately prune and burn heavily infected leaves.",
                "preventative": "<b>Preventative & Cultural Best Practices:</b>\n"
                            "• <b>Crop Spacing:</b> Maintain proper plant spacing to maximize solar radiation and canopy airflow.\n"
                            "• <b>Irrigation:</b> Switch to <b>Drip Irrigation</b> to avoid leaf wetness from overhead watering.\n"
                            "• <b>Nutrient Management:</b> Avoid excessive Nitrogen; apply balanced <b>Potassium & Organic Compost</b>.",
                "advice": "Prune infected leaves immediately, apply <b>Copper Oxychloride</b> or <b>Neem Oil</b>, and manage drip irrigation."
            }
