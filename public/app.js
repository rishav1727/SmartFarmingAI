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
    
    // Camera Elements
    const openCameraBtn = document.getElementById("open-camera-btn");
    const cameraModal = document.getElementById("camera-modal");
    const closeCameraBtn = document.getElementById("close-camera-btn");
    const cameraVideo = document.getElementById("camera-video");
    const snapBtn = document.getElementById("snap-btn");
    const cameraCanvas = document.getElementById("camera-canvas");
    let cameraStream = null;

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
    const nativeCameraInput = document.getElementById("native-camera-input");
    const browseGalleryBtn = document.getElementById("browse-gallery-btn");

    if (openCameraBtn && nativeCameraInput) {
        openCameraBtn.addEventListener("click", () => {
            nativeCameraInput.click();
        });
        nativeCameraInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFileSelection(e.target.files[0]);
                showToast("Photo captured successfully!", "info");
            }
        });
    }

    if (browseGalleryBtn && fileInput) {
        browseGalleryBtn.addEventListener("click", () => {
            fileInput.click();
        });
    }

    if (dropZone) {
        dropZone.addEventListener("click", (e) => {
            if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
                fileInput.click();
            }
        });
    }

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
            dropZone.classList.remove("hidden");
            previewContainer.classList.remove("hidden");
            analyzeBtn.classList.remove("disabled");
            analyzeBtn.disabled = false;

            // Automatically run AI Diagnostic as soon as photo is taken
            showToast("Photo captured! Running AI Health Diagnostic...", "info");
            setTimeout(() => {
                analyzeBtn.click();
            }, 300);
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = "";
        if (nativeCameraInput) nativeCameraInput.value = "";
        dropZone.classList.add("hidden");
        previewContainer.classList.add("hidden");
        imagePreview.src = "";
        analyzeBtn.classList.add("disabled");
        analyzeBtn.disabled = true;
    });// Camera Stream Handlers
    const switchCameraBtn = document.getElementById("switch-camera-btn");
    let currentFacingMode = "environment";

    async function startCamera(facingMode = "environment") {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
        }
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: facingMode }
            });
            cameraVideo.srcObject = cameraStream;
            cameraModal.classList.remove("hidden");
        } catch (err) {
            console.error("Camera access error:", err);
            showToast("Could not access device camera. Please allow camera permissions.", "error");
        }
    }

    if (openCameraBtn) {
        openCameraBtn.addEventListener("click", () => startCamera(currentFacingMode));
    }

    if (switchCameraBtn) {
        switchCameraBtn.addEventListener("click", () => {
            currentFacingMode = (currentFacingMode === "environment") ? "user" : "environment";
            startCamera(currentFacingMode);
        });
    }

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        cameraModal.classList.add("hidden");
    }

    if (closeCameraBtn) {
        closeCameraBtn.addEventListener("click", stopCamera);
    }

    if (snapBtn) {
        snapBtn.addEventListener("click", () => {
            if (!cameraVideo.videoWidth) return;
            
            cameraCanvas.width = cameraVideo.videoWidth;
            cameraCanvas.height = cameraVideo.videoHeight;
            const ctx = cameraCanvas.getContext("2d");
            ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
            
            cameraCanvas.toBlob((blob) => {
                if (blob) {
                    const cameraFile = new File([blob], `camera_leaf_${Date.now()}.jpg`, { type: "image/jpeg" });
                    handleFileSelection(cameraFile);
                    stopCamera();
                    showToast("Photo captured! Running diagnostic...", "info");
                    setTimeout(() => analyzeBtn.click(), 300);
                }
            }, "image/jpeg", 0.95);
        });
    }

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
            
            // Smooth scroll to diagnosis results card
            diagnosisCard.scrollIntoView({ behavior: "smooth", block: "start" });
            
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
    if (printBtn) {
        printBtn.addEventListener("click", () => {
            window.print();
        });
    }

    // Text-To-Speech Voice Assistant (Read Advice Out Loud)
    const readVoiceBtn = document.getElementById("read-voice-btn");
    let isSpeaking = false;
    if (readVoiceBtn) {
        readVoiceBtn.addEventListener("click", () => {
            if (!('speechSynthesis' in window)) {
                showToast("Voice playback is not supported on this browser.", "error");
                return;
            }

            if (isSpeaking) {
                window.speechSynthesis.cancel();
                isSpeaking = false;
                readVoiceBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i> Read Out Loud';
                return;
            }

            const activePanel = document.querySelector(".tab-panel.active");
            const textToRead = activePanel ? activePanel.innerText : "";
            if (!textToRead || textToRead.includes("No diagnosis")) {
                showToast("Please run a crop diagnosis first.", "info");
                return;
            }

            const utterance = new SpeechSynthesisUtterance(textToRead);
            utterance.lang = langSelect.value.toLowerCase() === "hindi" ? "hi-IN" : "en-US";
            utterance.rate = 0.95;

            utterance.onend = () => {
                isSpeaking = false;
                readVoiceBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i> Read Out Loud';
            };

            utterance.onerror = () => {
                isSpeaking = false;
                readVoiceBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i> Read Out Loud';
            };

            window.speechSynthesis.speak(utterance);
            isSpeaking = true;
            readVoiceBtn.innerHTML = '<i class="fa-solid fa-square"></i> Stop Voice';
            showToast(`Reading advice in ${langSelect.value}...`, "info");
        });
    }

    // WhatsApp Report Share
    const whatsappShareBtn = document.getElementById("whatsapp-share-btn");
    if (whatsappShareBtn) {
        whatsappShareBtn.addEventListener("click", () => {
            if (!currentDiagnosis) {
                showToast("Please run a crop diagnosis first.", "info");
                return;
            }

            const text = `🌿 *SmartFarming AI Diagnosis Report*\n` +
                         `===============================\n` +
                         `*Detected:* ${diseaseName.innerText}\n` +
                         `*Confidence:* ${confidencePercentage.innerText}\n\n` +
                         `*Overview:* ${document.getElementById("treatment-overview").innerText.slice(0, 150)}...\n\n` +
                         `*Chemical Control:* ${document.getElementById("treatment-chemical").innerText.slice(0, 150)}...\n\n` +
                         `Generated via SmartFarming AI Field Assistant`;
            
            const shareUrl = `https://api.whatsapp.com/send?text=${encodeURIComponent(text)}`;
            window.open(shareUrl, "_blank");
        });
    }

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

    // Plantix-Style Crop Pills Filter
    const cropPills = document.querySelectorAll(".crop-pill");
    cropPills.forEach(pill => {
        pill.addEventListener("click", () => {
            cropPills.forEach(p => p.classList.remove("active"));
            pill.classList.add("active");
            const cropType = pill.getAttribute("data-crop");
            if (cropType !== "all") {
                showToast(`Filter set to ${pill.innerText}. Take a photo of your ${cropType} leaf!`, "info");
            }
        });
    });

    // Plantix-Style Field Dosage Calculator
    const fieldAreaSelect = document.getElementById("field-area-select");
    const dosageResult = document.getElementById("dosage-result");

    if (fieldAreaSelect && dosageResult) {
        fieldAreaSelect.addEventListener("change", () => {
            const acres = parseFloat(fieldAreaSelect.value);
            const waterLiters = Math.round(acres * 200);
            const copperGrams = Math.round(acres * 200);
            const neemMl = Math.round(acres * 1000);

            dosageResult.innerHTML = `
                <i class="fa-solid fa-flask"></i> Estimated Dosage for <strong>${acres} Acre(s)</strong>:<br>
                • Chemical: <strong>${copperGrams}g Copper Oxychloride in ${waterLiters}L Water</strong><br>
                • Organic: <strong>${neemMl}ml Neem Oil Spray</strong>
            `;
        });
    // WhatsApp Modal Handlers
    const openWhatsappModalBtn = document.getElementById("open-whatsapp-modal-btn");
    const whatsappModal = document.getElementById("whatsapp-modal");
    const closeWhatsappBtn = document.getElementById("close-whatsapp-btn");

    if (openWhatsappModalBtn && whatsappModal) {
        openWhatsappModalBtn.addEventListener("click", () => {
            whatsappModal.classList.remove("hidden");
        });
    }

    if (closeWhatsappBtn && whatsappModal) {
        closeWhatsappBtn.addEventListener("click", () => {
            whatsappModal.classList.add("hidden");
        });
    }

    // Helper: Formatter for Disease Name
    function formatDiseaseName(rawName) {
        if (!rawName) return "";
        // Replace '___' with ': ' and underscores with spaces
        let formatted = rawName.replace("___", ": ").replace(/_/g, " ");
        // Capitalize first letters
        return formatted;
    }
});
