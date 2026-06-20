// SmartFarming.AI Dashboard Logic

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const removeBtn = document.getElementById("remove-btn");
    const analyzeBtn = document.getElementById("analyze-btn");
    
    const welcomePlaceholder = document.getElementById("welcome-placeholder");
    const diagnosisCard = document.getElementById("diagnosis-card");
    const explainCard = document.getElementById("explain-card");
    const treatmentCard = document.getElementById("treatment-card");
    
    const diseaseName = document.getElementById("disease-name");
    const confidencePercentage = document.getElementById("confidence-percentage");
    const confidenceBar = document.getElementById("confidence-bar");
    const viewUsed = document.getElementById("view-used");
    const energyScore = document.getElementById("energy-score");
    const top3List = document.getElementById("top3-list");
    
    const safetyWarning = document.getElementById("safety-warning");
    const safetyDesc = document.getElementById("safety-desc");
    
    const baseImage = document.getElementById("base-image");
    const heatmapImage = document.getElementById("heatmap-image");
    const opacitySlider = document.getElementById("opacity-slider");
    
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanels = document.querySelectorAll(".tab-panel");
    const langSelect = document.getElementById("lang-select");
    const indexDocsBtn = document.getElementById("index-docs-btn");
    
    // Chat Elements
    const chatDrawer = document.getElementById("chat-drawer");
    const chatToggle = document.getElementById("chat-toggle");
    const chatMessages = document.getElementById("chat-messages");
    const chatInput = document.getElementById("chat-input");
    const chatSendBtn = document.getElementById("chat-send-btn");
    const clearChat = document.getElementById("clear-chat");
    const chatBadge = document.getElementById("chat-badge");
    
    // Global Loader
    const globalLoader = document.getElementById("global-loader");
    const loaderText = document.getElementById("loader-text");
    
    // State variables
    let selectedFile = null;
    let currentDiagnosis = null;
    let chatHistory = [];
    
    // 1. Toast Notifications System
    function showToast(message, type = "success") {
        const container = document.getElementById("toast-container");
        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        
        let icon = '<i class="fa-solid fa-circle-check"></i>';
        if (type === "error") {
            icon = '<i class="fa-solid fa-circle-xmark"></i>';
        } else if (type === "info") {
            icon = '<i class="fa-solid fa-circle-info"></i>';
        }
        
        toast.innerHTML = `${icon}<span>${message}</span>`;
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateX(50px)";
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
    
    // 2. Global Loader helper
    function setLoader(show, text = "Analyzing...") {
        loaderText.innerText = text;
        if (show) {
            globalLoader.classList.remove("hidden");
        } else {
            globalLoader.classList.add("hidden");
        }
    }

    // 3. Drop Zone & File Selection
    dropZone.addEventListener("click", (e) => {
        if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener("change", (e) => {
        handleFileSelection(e.target.files[0]);
    });

    // Drag-and-drop events
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add("hover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove("hover");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        handleFileSelection(dt.files[0]);
    });

    function handleFileSelection(file) {
        if (!file) return;
        
        if (!file.type.startsWith("image/")) {
            showToast("Invalid file type. Please upload an image file.", "error");
            return;
        }
        
        selectedFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove("hidden");
            analyzeBtn.classList.remove("disabled");
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = "";
        previewContainer.classList.add("hidden");
        imagePreview.src = "";
        analyzeBtn.classList.add("disabled");
        analyzeBtn.disabled = true;
    });

    // 4. Run Diagnosis Pipeline
    analyzeBtn.addEventListener("click", async () => {
        if (!selectedFile) return;
        
        setLoader(true, "Detecting crop health via Vision Transformer...");
        
        const formData = new FormData();
        formData.append("file", selectedFile);
        
        try {
            // Step 1: Detect & Explain (FastAPI endpoint runs inference & Grad-CAM)
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Server inference error");
            }
            
            const result = await response.json();
            currentDiagnosis = result;
            
            // Render basic details
            diseaseName.innerText = formatDiseaseName(result.disease);
            confidencePercentage.innerText = `${result.confidence.toFixed(1)}%`;
            confidenceBar.style.width = `${result.confidence}%`;
            viewUsed.innerText = result.view_used;
            energyScore.innerText = result.energy_score.toFixed(2);
            
            // Handle OOD safety fallbacks
            if (result.is_ood) {
                safetyWarning.classList.remove("hidden");
                if (result.detected_object) {
                    safetyDesc.innerText = `This image looks like: '${result.detected_object}'. The AI system is trained strictly on plant crop diseases.`;
                } else {
                    safetyDesc.innerText = "This image is highly anomalous and does not contain recognizable plant foliage.";
                }
                showToast("Out-of-Distribution anomaly blocked for safety.", "info");
            } else {
                safetyWarning.classList.add("hidden");
            }
            
            // Load images in visualizer
            baseImage.src = result.image_url;
            if (result.heatmap_url) {
                heatmapImage.src = result.heatmap_url;
                explainCard.classList.remove("hidden");
            } else {
                explainCard.classList.add("hidden");
            }
            
            // Update Top 3 candidates list
            top3List.innerHTML = "";
            result.top3.forEach(candidate => {
                const li = document.createElement("li");
                li.className = "top3-item";
                li.innerHTML = `
                    <span class="top-name">${formatDiseaseName(candidate.disease)}</span>
                    <span class="top-pct">${candidate.confidence.toFixed(1)}%</span>
                `;
                top3List.appendChild(li);
            });
            
            // Enable dashboard visibility
            welcomePlaceholder.classList.add("hidden");
            diagnosisCard.classList.remove("hidden");
            treatmentCard.classList.remove("hidden");
            chatDrawer.classList.remove("hidden");
            
            // Step 2: Cure (Get LLM advice in chosen language)
            setLoader(true, "Generating custom treatment guidelines via LLM...");
            await fetchAdvice();
            
            // Enable Chat Interface
            chatInput.disabled = false;
            chatSendBtn.disabled = false;
            chatBadge.classList.remove("hidden");
            
            showToast("Analysis complete. Treatment advice generated!");
            
        } catch (err) {
            console.error("Diagnosis Error:", err);
            showToast(err.message || "Failed to analyze leaf image.", "error");
        } finally {
            setLoader(false);
        }
    });

    // 5. Fetch Advice (Cure step)
    async function fetchAdvice() {
        if (!currentDiagnosis) return;
        
        const language = langSelect.value;
        try {
            const response = await fetch("/api/advice", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    disease: currentDiagnosis.disease,
                    confidence: currentDiagnosis.confidence,
                    language: language
                })
            });
            
            if (!response.ok) {
                throw new Error("Failed to fetch advice");
            }
            
            const advice = await response.json();
            
            // Update Tab panels
            document.getElementById("treatment-overview").innerText = advice.overview || "Overview not available.";
            document.getElementById("treatment-chemical").innerText = advice.chemical || "Chemical control recommendations not available.";
            document.getElementById("treatment-biological").innerText = advice.biological || "Biological/organic control recommendations not available.";
            document.getElementById("treatment-preventative").innerText = advice.preventative || "Preventative/cultural recommendations not available.";
            
        } catch (err) {
            console.error("Advice Error:", err);
            showToast("Failed to fetch custom treatment advice.", "error");
        }
    }

    // Language selector change listener
    langSelect.addEventListener("change", async () => {
        if (currentDiagnosis) {
            setLoader(true, `Translating guidelines to ${langSelect.value}...`);
            await fetchAdvice();
            setLoader(false);
            showToast(`Advice language updated to ${langSelect.value}`);
        }
    });

    // 6. Blending Opacity Slider
    opacitySlider.addEventListener("input", (e) => {
        const val = e.target.value;
        heatmapImage.style.opacity = val / 100;
    });

    const printBtn = document.getElementById("print-btn");
    printBtn.addEventListener("click", () => {
        window.print();
    });

    // 7. Tabs Functionality
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanels.forEach(p => p.classList.remove("active"));
            
            btn.classList.add("active");
            const targetPanel = document.getElementById(btn.getAttribute("data-tab"));
            targetPanel.classList.add("active");
        });
    });

    // 8. Sync Documents / RAG
    indexDocsBtn.addEventListener("click", async () => {
        setLoader(true, "Syncing local database and compiling RAG embeddings...");
        try {
            const response = await fetch("/api/index-docs", { method: "POST" });
            const result = await response.json();
            
            if (response.ok) {
                showToast(result.message);
            } else {
                showToast(result.message || "No documents found to sync.", "info");
            }
        } catch (err) {
            console.error("RAG Sync Error:", err);
            showToast("Failed to sync local database.", "error");
        } finally {
            setLoader(false);
        }
    });

    // 9. Upload Document to RAG
    const docUploadInput = document.getElementById("doc-upload-input");
    docUploadInput.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        setLoader(true, `Uploading and indexing '${file.name}' into RAG...`);
        
        const formData = new FormData();
        formData.append("file", file);
        
        try {
            const response = await fetch("/api/documents/upload", {
                method: "POST",
                body: formData
            });
            const result = await response.json();
            
            if (response.ok) {
                showToast(result.message, "success");
            } else {
                showToast(result.detail || "Failed to upload reference document.", "error");
            }
        } catch (err) {
            console.error("Document Upload Error:", err);
            showToast("Network error uploading document.", "error");
        } finally {
            docUploadInput.value = "";
            setLoader(false);
        }
    });

    // 9. Floating Chatbot Interaction
    chatToggle.addEventListener("click", () => {
        chatDrawer.classList.toggle("collapsed");
        chatBadge.classList.add("hidden");
    });

    chatSendBtn.addEventListener("click", () => {
        sendChatMessage();
    });

    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            sendChatMessage();
        }
    });

    async function sendChatMessage() {
        const text = chatInput.value.trim();
        if (!text || !currentDiagnosis) return;
        
        // Add User Message
        appendMessage("user", text);
        chatInput.value = "";
        
        // Disable input while bot replies
        chatInput.disabled = true;
        chatSendBtn.disabled = true;
        
        // Add Loading bubble
        const botMsgDiv = document.createElement("div");
        botMsgDiv.className = "message bot typing";
        botMsgDiv.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Thinking...';
        chatMessages.appendChild(botMsgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    disease: currentDiagnosis.disease,
                    history: chatHistory,
                    message: text,
                    language: langSelect.value
                })
            });
            
            botMsgDiv.remove();
            
            if (!response.ok) {
                throw new Error("Chat error");
            }
            
            const data = await response.json();
            appendMessage("bot", data.reply);
            
        } catch (err) {
            botMsgDiv.remove();
            console.error("Chat Error:", err);
            appendMessage("bot", "Sorry, I am having trouble connecting right now. Please try again.");
        } finally {
            chatInput.disabled = false;
            chatSendBtn.disabled = false;
            chatInput.focus();
        }
    }

    function appendMessage(sender, text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender}`;
        msgDiv.innerText = text;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Save to state history
        chatHistory.push({ sender, text });
    }

    clearChat.addEventListener("click", () => {
        chatMessages.innerHTML = "";
        chatHistory = [];
        showToast("Conversation cleared.", "info");
    });

    // Helper: Formatter for Disease Name
    function formatDiseaseName(rawName) {
        if (!rawName) return "";
        // Replace '___' with ': ' and underscores with spaces
        let formatted = rawName.replace("___", ": ").replace(/_/g, " ");
        // Capitalize first letters
        return formatted;
    }
});
