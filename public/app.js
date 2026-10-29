// SmartFarming.AI Dashboard Logic
const API_BASE = (window.location.hostname.includes('vercel.app') || (window.location.hostname === 'localhost' && window.location.port !== '8000'))
    ? 'https://smartfarming-ai.onrender.com'
    : '';

function resolveUrl(url) {
    if (!url) return url;
    if (url.startsWith('http://') || url.startsWith('https://')) return url;
    return API_BASE + url;
}


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

    // 2b. Safe JSON parser — prevents "Unexpected token '<'" when server returns HTML error page
    async function safeJson(response) {
        const text = await response.text();
        try {
            return JSON.parse(text);
        } catch {
            if (text.trim().startsWith("<")) {
                throw new Error("Server is warming up. The AI engine loads on first use — please try again in a few seconds.");
            }
            throw new Error("Invalid server response: " + text.slice(0, 120));
        }
    }

    // 3. Drop Zone & File Selection
    const nativeCameraInput = document.getElementById("native-camera-input");
    const browseGalleryBtn = document.getElementById("browse-gallery-btn");
    const dropZonePrompt = document.getElementById("drop-zone-prompt");

    if (nativeCameraInput) {
        nativeCameraInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFileSelection(e.target.files[0]);
                showToast("Photo captured successfully!", "info");
            }
        });
    }

    if (browseGalleryBtn && fileInput) {
        browseGalleryBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    if (dropZone) {
        dropZone.addEventListener("click", (e) => {
            if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
                if (fileInput) fileInput.click();
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFileSelection(e.target.files[0]);
            }
        });
    }

    // Drag-and-drop events
    if (dropZone) {
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
            if (dt && dt.files && dt.files[0]) {
                handleFileSelection(dt.files[0]);
            }
        });
    }

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
            if (dropZonePrompt) dropZonePrompt.classList.add("hidden");
            if (previewContainer) previewContainer.classList.remove("hidden");
            analyzeBtn.classList.remove("disabled");
            analyzeBtn.disabled = false;

            // Automatically run AI Diagnostic as soon as photo is taken
            showToast("Photo loaded! Running AI Health Diagnostic...", "info");
            setTimeout(() => {
                analyzeBtn.click();
            }, 300);
        };
        reader.readAsDataURL(file);
    }

    // Sample Chip Buttons 1-Click Test Handler
    const sampleBtns = document.querySelectorAll(".sample-chip-btn");
    sampleBtns.forEach(btn => {
        btn.addEventListener("click", async (e) => {
            e.stopPropagation();
            const url = btn.getAttribute("data-url");
            try {
                setLoader(true, "Fetching sample image...");
                const response = await fetch(url);
                const blob = await response.blob();
                const file = new File([blob], url.split('/').pop(), { type: "image/jpeg" });
                setLoader(false);
                handleFileSelection(file);
            } catch (err) {
                setLoader(false);
                showToast("Failed to load sample image: " + err.message, "error");
            }
        });
    });

    if (removeBtn) {
        removeBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            selectedFile = null;
            if (fileInput) fileInput.value = "";
            if (nativeCameraInput) nativeCameraInput.value = "";
            if (dropZonePrompt) dropZonePrompt.classList.remove("hidden");
            if (previewContainer) previewContainer.classList.add("hidden");
            imagePreview.src = "";
            analyzeBtn.classList.add("disabled");
            analyzeBtn.disabled = true;
        });
    }

    // Camera Stream Handlers
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
            // Fallback to native camera input or file picker if getUserMedia fails
            if (nativeCameraInput) {
                nativeCameraInput.click();
            } else if (fileInput) {
                fileInput.click();
            } else {
                showToast("Could not access device camera. Please allow camera permissions.", "error");
            }
        }
    }

    if (openCameraBtn) {
        openCameraBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            startCamera(currentFacingMode);
        });
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
        if (selectedCrop && selectedCrop !== "all") {
            formData.append("crop", selectedCrop);
        }
        
        try {
            // Step 1: Detect & Explain (FastAPI endpoint runs inference & Grad-CAM)
            const response = await fetch(`${API_BASE}/api/predict`, {
                method: "POST",
                body: formData
            });
            
            if (!response.ok) {
                const contentType = response.headers.get("content-type") || "";
                if (contentType.includes("application/json")) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || "Server inference error");
                } else {
                    // HTML 502/503/404 page returned (e.g. Render spin-up or gateway timeout)
                    throw new Error("AI server is initializing or warming up. Please wait 10 seconds and try again.");
                }
            }
            
            const result = await safeJson(response);
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
                const obj = result.detected_object ? result.detected_object : "";
                const plantFoods = ["cabbage", "broccoli", "cauliflower", "lettuce", "spinach", "kale",
                    "tomato", "potato", "pepper", "corn", "wheat", "rice", "apple", "grape", "strawberry",
                    "peach", "cherry", "orange", "banana", "mango", "leaf", "plant", "flower", "vegetable",
                    "fruit", "crop", "herb", "tree", "mushroom", "onion", "garlic", "carrot", "bean", "pea"];
                const objLower = obj.toLowerCase();
                const looksLikePlant = plantFoods.some(p => objLower.includes(p));
                
                if (looksLikePlant && obj) {
                    safetyDesc.innerText = `Detected: '${obj}'. This appears to be a plant/food item but doesn't match our 38 trained disease categories. Try selecting the specific crop above and re-upload.`;
                } else if (obj) {
                    safetyDesc.innerText = `This image looks like: '${obj}'. Please upload a clear, close-up photo of a plant leaf with visible disease symptoms.`;
                } else {
                    safetyDesc.innerText = "Image quality or content could not be confirmed as a plant leaf. Please upload a clear, close-up photo showing disease symptoms on a leaf.";
                }
                showToast("Please upload a clear plant leaf image for best results.", "info");
            } else {
                safetyWarning.classList.add("hidden");
            }
            
            // Load images in visualizer
            baseImage.src = resolveUrl(result.image_url);
            if (result.heatmap_url) {
                heatmapImage.src = resolveUrl(result.heatmap_url);
                explainCard.classList.remove("hidden");
                triggerHeatmapScanAnimation();
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
            const response = await fetch(`${API_BASE}/api/advice`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    disease: currentDiagnosis.disease,
                    confidence: currentDiagnosis.confidence,
                    language: language
                })
            });
            
            if (!response.ok) {
                throw new Error("Treatment advice server error. Ensure the Python backend is running.");
            }
            
            const advice = await safeJson(response);
            
            function formatAdviceHTML(text) {
                if (!text) return "";
                let formatted = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                    .replace(/__(.*?)__/g, '<strong>$1</strong>');
                formatted = formatted.replace(/\n/g, '<br>');
                return formatted;
            }
            
            // Update Tab panels with bold HTML support
            document.getElementById("treatment-overview").innerHTML = formatAdviceHTML(advice.overview) || "Overview not available.";
            document.getElementById("treatment-chemical").innerHTML = formatAdviceHTML(advice.chemical) || "Chemical control recommendations not available.";
            document.getElementById("treatment-biological").innerHTML = formatAdviceHTML(advice.biological) || "Biological/organic control recommendations not available.";
            document.getElementById("treatment-preventative").innerHTML = formatAdviceHTML(advice.preventative) || "Preventative/cultural recommendations not available.";
            
        } catch (err) {
            console.error("Advice Error:", err);
            showToast("Failed to fetch custom treatment advice.", "error");
        }
    }

    // ==========================================
    // 5b. HEATMAP ANIMATION & CONTROLS MODULE
    // ==========================================
    const heatmapAutoBtn = document.getElementById("heatmap-auto-btn");
    const heatmapFreezeBtn = document.getElementById("heatmap-freeze-btn");
    const heatmapRescanBtn = document.getElementById("heatmap-rescan-btn");
    const aiScanBeam = document.getElementById("ai-scan-beam");
    const heatmapScanLabel = document.getElementById("heatmap-scan-label");
    const heatmapScanText = document.getElementById("heatmap-scan-text");
    const hotspotMarkers = document.getElementById("hotspot-markers");
    const heatmapComparison = document.getElementById("heatmap-comparison");

    if (opacitySlider && heatmapImage) {
        opacitySlider.addEventListener("input", (e) => {
            if (heatmapImage.classList.contains("auto-pulsing")) {
                heatmapImage.classList.remove("auto-pulsing");
                if (heatmapAutoBtn) heatmapAutoBtn.classList.remove("active");
                if (heatmapFreezeBtn) heatmapFreezeBtn.classList.add("active");
            }
            heatmapImage.style.opacity = e.target.value / 100;
        });
    }

    function setHeatmapMode(mode) {
        if (!heatmapImage) return;
        if (mode === "auto") {
            heatmapImage.classList.add("auto-pulsing");
            if (heatmapAutoBtn) heatmapAutoBtn.classList.add("active");
            if (heatmapFreezeBtn) heatmapFreezeBtn.classList.remove("active");
        } else if (mode === "freeze") {
            heatmapImage.classList.remove("auto-pulsing");
            if (opacitySlider) heatmapImage.style.opacity = opacitySlider.value / 100;
            if (heatmapAutoBtn) heatmapAutoBtn.classList.remove("active");
            if (heatmapFreezeBtn) heatmapFreezeBtn.classList.add("active");
        }
    }

    if (heatmapAutoBtn) {
        heatmapAutoBtn.addEventListener("click", () => setHeatmapMode("auto"));
    }
    if (heatmapFreezeBtn) {
        heatmapFreezeBtn.addEventListener("click", () => setHeatmapMode("freeze"));
    }
    if (heatmapRescanBtn) {
        heatmapRescanBtn.addEventListener("click", () => triggerHeatmapScanAnimation());
    }

    function triggerHeatmapScanAnimation() {
        if (!aiScanBeam || !heatmapScanLabel) return;

        // Reset beam & markers
        aiScanBeam.classList.remove("scanning");
        heatmapScanLabel.classList.remove("visible");
        if (heatmapImage) heatmapImage.classList.remove("auto-pulsing");
        if (hotspotMarkers) hotspotMarkers.innerHTML = "";

        // Trigger scan beam
        void aiScanBeam.offsetWidth; // reflow trigger
        aiScanBeam.classList.add("scanning");
        heatmapScanLabel.classList.add("visible");
        if (heatmapScanText) heatmapScanText.innerText = "Vision Transformer Scanning Leaf...";

        // Step 1: Status text update mid-scan
        setTimeout(() => {
            if (heatmapScanText) heatmapScanText.innerText = "Computing Grad-CAM Attention Map...";
        }, 1000);

        // Step 2: Scan completes — flash overlay & spawn hotspot pulse targets
        setTimeout(() => {
            if (heatmapScanText) heatmapScanText.innerText = "Infection Hotspots Identified!";
            
            // Add flash effect
            if (heatmapComparison) {
                const flash = document.createElement("div");
                flash.className = "heatmap-flash";
                heatmapComparison.appendChild(flash);
                setTimeout(() => flash.remove(), 600);
            }

            // Inject 3 disease hotspot targets
            if (hotspotMarkers) {
                const spots = [
                    { top: "35%", left: "48%" },
                    { top: "52%", left: "62%" },
                    { top: "68%", left: "38%" }
                ];
                spots.forEach((pos, idx) => {
                    setTimeout(() => {
                        const spot = document.createElement("div");
                        spot.className = "hotspot visible";
                        spot.style.top = pos.top;
                        spot.style.left = pos.left;
                        spot.innerHTML = `
                            <div class="hotspot-ring"></div>
                            <div class="hotspot-ring"></div>
                            <div class="hotspot-ring"></div>
                            <div class="hotspot-core"></div>
                        `;
                        hotspotMarkers.appendChild(spot);
                    }, idx * 250);
                });
            }

            // Start auto pulse mode
            setHeatmapMode("auto");

        }, 2200);

        // Step 3: Hide scan status badge
        setTimeout(() => {
            heatmapScanLabel.classList.remove("visible");
        }, 4500);
    }

    // Dynamic Client-side UI i18n Translation Dictionary
    const UI_TRANSLATIONS = {
        Hindi: {
            logoSub: "प्लांटिक्स-स्टाइल फसल रोग निदान और कृषि परामर्श",
            selectCropLabel: "<i class=\"fa-solid fa-wheat-field\"></i> फसल चुनें:",
            cropAll: "<i class=\"fa-solid fa-seedling\"></i> सभी फसलें",
            cropApple: "🍎 सेब (Apple)",
            cropPotato: "🥔 आलू (Potato)",
            cropTomato: "🍅 टमाटर (Tomato)",
            cropGrape: "🍇 अंगूर (Grape)",
            cropCorn: "🌽 मक्का (Corn)",
            cropRice: "🌾 चावल (Rice)",
            cropWheat: "🌾 गेहूं (Wheat)",
            sampleApple: "🍎 सेब रोग (Apple Scab)",
            samplePotato: "🥔 आलू झुलसा (Potato Blight)",
            sampleTomato: "🍅 टमाटर फफूंद (Tomato Mold)",
            sampleCorn: "🌽 मक्का रस्ट (Corn Rust)",
            step1Title: "पत्ते की फोटो लें",
            step1Sub: "बीमार पत्ते की साफ़ तस्वीर खींचें",
            step2Title: "एआई हीटमैप जांच",
            step2Sub: "Grad-CAM संक्रमित हिस्सा पहचानेगा",
            step3Title: "3-चरण उपचार उपाय",
            step3Sub: "जैविक, रासायनिक व वॉयस सलाह पाएँ",
            plantixTitle: "प्लांटिक्स एआई क्रॉप डॉक्टर",
            plantixSub: "पौधों के रोगों का तुरंत निदान और सटीक उपचार पाएं",
            cameraBtnText: "बीमार पत्ते की फोटो खींचें",
            cameraBtnSub: "फोन का एचडी कैमरा अपने आप खुलेगा",
            galleryBtnText: "<i class=\"fa-solid fa-images\"></i> गैलरी से फोटो चुनें",
            samplesTitle: "<i class=\"fa-solid fa-vial\"></i> या 1-क्लिक क्विक टेस्ट सैंपल आज़माएं:",
            dropInstructions: "यहाँ पत्ते की फोटो खींचकर लाएं या चुनें",
            dropFormats: "JPG, PNG, WEBP सपोर्टेड (अधिकतम 10MB)",
            analyzeBtn: "<i class=\"fa-solid fa-stethoscope\"></i> एआई रोग निदान चलाएं",
            oodTitle: "गैर-फसल फोटो की पहचान",
            oodDesc: "यह तस्वीर कृषि फसल के पत्ते से मेल नहीं खाती।",
            diagTitle: "<i class=\"fa-solid fa-square-poll-vertical\"></i> फसल निदान रिपोर्ट",
            confLabel: "सटीकता दर",
            tabOverview: "<i class=\"fa-solid fa-file-medical\"></i> मुख्य विवरण",
            tabChemical: "<i class=\"fa-solid fa-vial-circle-check\"></i> रासायनिक रोकथाम",
            tabBiological: "<i class=\"fa-solid fa-seedling\"></i> जैविक / ऑर्गेनिक उपचार",
            tabPreventative: "<i class=\"fa-solid fa-shield-halved\"></i> बचाव के उपाय",
            dosageTitle: "खेत छिड़काव मात्रा कैलकुलेटर",
            fieldAreaLabel: "खेत का क्षेत्रफल:",
            voiceBtn: "<i class=\"fa-solid fa-volume-high\"></i> बोलकर सुनें",
            whatsappBtn: "<i class=\"fa-brands fa-whatsapp\"></i> व्हाट्सएप पर शेयर करें",
            printBtn: "<i class=\"fa-solid fa-print\"></i> रिपोर्ट प्रिंट करें",
            heatmapTitle: "<i class=\"fa-solid fa-bullseye\"></i> एआई डिटेक्शन हीटमैप (Grad-CAM)",
            heatmapSub: "संक्रमित पत्तों का विश्लेषण (ViT विज़न)",
            opacityLabel: "हीटमैप पारदर्शिता:",
            chatTitle: "<i class=\"fa-solid fa-robot\"></i> किसान एआई चैट सलाहकार",
            chatSub: "दवा, छिड़काव, मिट्टी या सुरक्षा पर सवाल पूछें",
            chatPlaceholder: "अपना सवाल यहाँ लिखें...",
            sendBtn: "<i class=\"fa-solid fa-paper-plane\"></i> भेजें",
            navHome: "होम",
            navCamera: "कैमरा",
            navDiagnosis: "रिपोर्ट",
            navWhatsapp: "व्हाट्सएप",
            disclaimer: "एआई और कृषि विशेषज्ञों द्वारा सत्यापित सिफारिशें। छिड़काव से पहले local डोज चेक करें।"
        },
        Bengali: {
            logoSub: "প্ল্যান্টিক্স-স্টাইল ফসল রোগ নির্ণয় ও পরামর্শ",
            selectCropLabel: "<i class=\"fa-solid fa-wheat-field\"></i> ফসল বাছুন:",
            cropAll: "<i class=\"fa-solid fa-seedling\"></i> সব ফসল",
            cropApple: "🍎 আপেল",
            cropPotato: "🥔 আলু",
            cropTomato: "🍅 টমেটো",
            cropGrape: "🍇 আঙুর",
            cropCorn: "🌽 ভুট্টা",
            cropRice: "🌾 ধান",
            cropWheat: "🌾 গম",
            sampleApple: "🍎 আপেল স্কেব",
            samplePotato: "🥔 আলু মরক",
            sampleTomato: "🍅 টমেটো মোল্ড",
            sampleCorn: "🌽 ভুট্টা রাস্ট",
            step1Title: "পাতার ছবি তুলুন",
            step1Sub: "আক্রান্ত পাতার স্পষ্ট ছবি তুলুন",
            step2Title: "এআই হিটম্যাপ পরীক্ষা",
            step2Sub: "Grad-CAM আক্রান্ত অংশ সনাক্ত করবে",
            step3Title: "৩-ধাপের চিকিৎসা",
            step3Sub: "জৈব, রাসায়নিক ও ভয়েস পরামর্শ পান",
            plantixTitle: "প্ল্যান্টিক্স এআই ক্রপ ডাক্তার",
            plantixSub: "গাছের রোগ নির্ণয় ও তাৎক্ষণিক সঠিক চিকিৎসা",
            cameraBtnText: "আক্রান্ত পাতার ছবি তুলুন",
            cameraBtnSub: "ফোনের ক্যামেরা স্বয়ংক্ৰিয়ভাবে খুলবে",
            galleryBtnText: "<i class=\"fa-solid fa-images\"></i> গ্যালারি থেকে ছবি বাছুন",
            samplesTitle: "<i class=\"fa-solid fa-vial\"></i> বা ১-ক্লিক কুইক টেস্ট স্যাম্পল পরীক্ষা করুন:",
            dropInstructions: "এখানে পাতার ছবি এনে রাখুন বা ক্লিক করুন",
            dropFormats: "JPG, PNG, WEBP সমর্থিত",
            analyzeBtn: "<i class=\"fa-solid fa-stethoscope\"></i> এআই রোগ পরীক্ষা করুন",
            diagTitle: "<i class=\"fa-solid fa-square-poll-vertical\"></i> ফসল নির্ণয় রিপোর্ট",
            tabOverview: "<i class=\"fa-solid fa-file-medical\"></i> বিবরণ",
            tabChemical: "<i class=\"fa-solid fa-vial-circle-check\"></i> রাসায়নিক প্রতিরোধ",
            tabBiological: "<i class=\"fa-solid fa-seedling\"></i> জৈব চিকিৎসা",
            tabPreventative: "<i class=\"fa-solid fa-shield-halved\"></i> প্রতিরোধমূলক ব্যবস্থা",
            dosageTitle: "কীটনাশক মাত্রা ক্যালকুলেটর",
            fieldAreaLabel: "জমির পরিমাণ:",
            voiceBtn: "<i class=\"fa-solid fa-volume-high\"></i> শুনে নিন",
            whatsappBtn: "<i class=\"fa-brands fa-whatsapp\"></i> হোয়াটসঅ্যাপে শেয়ার করুন",
            printBtn: "<i class=\"fa-solid fa-print\"></i> প্রিন্ট রিপোর্ট",
            chatTitle: "<i class=\"fa-solid fa-robot\"></i> কিষাণ এআই চ্যাট সহায়ক",
            chatPlaceholder: "আপনার প্রশ্ন এখানে লিখুন...",
            navHome: "হোম",
            navCamera: "ক্যামেরা",
            navDiagnosis: "রিপোর্ট",
            navWhatsapp: "হোয়াটসঅ্যাপ"
        },
        Telugu: {
            logoSub: "ప్లాంటిక్స్-స్టైల్ పంట వ్యాధి నిర్ధారణ & వ్యవసాయ సలహా",
            selectCropLabel: "<i class=\"fa-solid fa-wheat-field\"></i> పంటను ఎంచుకోండి:",
            cropAll: "<i class=\"fa-solid fa-seedling\"></i> అన్ని పంటలు",
            cropApple: "🍎 యాపిల్",
            cropPotato: "🥔 బంగాళాదుంప",
            cropTomato: "🍅 టమాటా",
            cropGrape: "🍇 ద్రాక్ష",
            cropCorn: "🌽 జొన్న/మొక్కజొన్న",
            cropRice: "🌾 వరి",
            cropWheat: "🌾 గోధుమ",
            sampleApple: "🍎 యాపిల్ స్కాబ్",
            samplePotato: "🥔 ఆలు లేట్ బ్లైట్",
            sampleTomato: "🍅 టమాటా లీఫ్ మోల్డ్",
            sampleCorn: "🌽 మొక్కజొన్న రస్ట్",
            step1Title: "ఆకు ఫోటో తీయండి",
            step1Sub: "వ్యాధి సోకిన ఆకు ఫోటోను తీయండి",
            step2Title: "AI హీట్‌మ్యాప్ తనిఖీ",
            step2Sub: "Grad-CAM వ్యాధి ప్రభావాన్ని గుర్తిస్తుంది",
            step3Title: "3-అంచెల చికిత్స",
            step3Sub: "రసాయన, సేంద్రీయ మరియు వాయిస్ సలహా",
            plantixTitle: "ప్లాంటిక్స్ AI క్రాప్ డాక్టర్",
            plantixSub: "పంట రోగ నిర్ధారణ మరియు తక్షణ నివారణ చర్యలు",
            cameraBtnText: "ఆకు ఫోటో తీయండి",
            cameraBtnSub: "మీ ఫోన్ కెమెరా స్వయంచాలకంగా తెరుచుకుంటుంది",
            galleryBtnText: "<i class=\"fa-solid fa-images\"></i> గ్యాలరీ నుండి ఎంచుకోండి",
            samplesTitle: "<i class=\"fa-solid fa-vial\"></i> లేదా 1-క్లిక్ టెస్ట్ శాంపిల్స్ చూడండి:",
            analyzeBtn: "<i class=\"fa-solid fa-stethoscope\"></i> AI వ్యాధి నిర్ధారణ చేయండి",
            diagTitle: "<i class=\"fa-solid fa-square-poll-vertical\"></i> పంట రోగ నివేదిక",
            tabOverview: "<i class=\"fa-solid fa-file-medical\"></i> అవలోకనం",
            tabChemical: "<i class=\"fa-solid fa-vial-circle-check\"></i> రసాయన నివారణ",
            tabBiological: "<i class=\"fa-solid fa-seedling\"></i> సేంద్రీయ నివారణ",
            tabPreventative: "<i class=\"fa-solid fa-shield-halved\"></i> ముందస్తు జాగ్రత్తలు",
            dosageTitle: "మందుల మోతాదు క్యాలిక్యులేటర్",
            voiceBtn: "<i class=\"fa-solid fa-volume-high\"></i> వినండి",
            whatsappBtn: "<i class=\"fa-brands fa-whatsapp\"></i> వాట్సాప్‌లో షేర్ చేయండి",
            chatTitle: "<i class=\"fa-solid fa-robot\"></i> కిసాన్ AI చాట్ అసిస్టెంట్",
            chatPlaceholder: "మీ ప్రశ్నను ఇక్కడ టైప్ చేయండి...",
            navHome: "హోమ్",
            navCamera: "కెమెరా",
            navDiagnosis: "రిపోర్ట్",
            navWhatsapp: "వాట్సాప్"
        },
        Marathi: {
            logoSub: "प्लँटिक्स-स्टाईल पीक रोग निदान आणि कृषी सल्ला",
            selectCropLabel: "<i class=\"fa-solid fa-wheat-field\"></i> पीक निवडा:",
            cropAll: "<i class=\"fa-solid fa-seedling\"></i> सर्व पिके",
            cropApple: "🍎 सफरचंद",
            cropPotato: "🥔 बटाटा",
            cropTomato: "🍅 टोमॅटो",
            cropGrape: "🍇 द्राक्षे",
            cropCorn: "🌽 मका",
            cropRice: "🌾 भात",
            cropWheat: "🌾 गवत/गहू",
            sampleApple: "🍎 सफरचंद स्कॅब",
            samplePotato: "🥔 बटाटा करपा",
            sampleTomato: "🍅 टोमॅटो बुरशी",
            sampleCorn: "🌽 मका तांबेरा",
            step1Title: "पानाचा फोटो घ्या",
            step1Sub: "रोगट पानाचा स्पष्ट फोटो काढा",
            step2Title: "AI हीटमॅप तपासणी",
            step2Sub: "Grad-CAM प्रभावित भाग शोधेल",
            step3Title: "3-टप्प्यांची औषधोपचार",
            step3Sub: "जैविक, रासायनिक व व्हॉइस सल्ला मिळवा",
            plantixTitle: "प्लँटिक्स AI पीक डॉक्टर",
            plantixSub: "पिकांच्या रोगांचे अचूक निदान व झटपट उपचार",
            cameraBtnText: "पानाचा फोटो काढा",
            cameraBtnSub: "तुमचा मोबाईल कॅमेरा आपोआप उघडेल",
            galleryBtnText: "<i class=\"fa-solid fa-images\"></i> गॅलरीतून निवडा",
            samplesTitle: "<i class=\"fa-solid fa-vial\"></i> किंवा 1-क्लिक टेस्ट सॅम्पल तपासा:",
            analyzeBtn: "<i class=\"fa-solid fa-stethoscope\"></i> AI रोग तपासणी सुरू करा",
            diagTitle: "<i class=\"fa-solid fa-square-poll-vertical\"></i> पीक रोग निदान अहवाल",
            tabOverview: "<i class=\"fa-solid fa-file-medical\"></i> माहिती",
            tabChemical: "<i class=\"fa-solid fa-vial-circle-check\"></i> रासायनिक नियंत्रण",
            tabBiological: "<i class=\"fa-solid fa-seedling\"></i> जैविक नियंत्रण",
            tabPreventative: "<i class=\"fa-solid fa-shield-halved\"></i> प्रतिबंधात्मक उपाय",
            dosageTitle: "फवारणी प्रमाण कॅल्क्युलेटर",
            voiceBtn: "<i class=\"fa-solid fa-volume-high\"></i> ऐका",
            whatsappBtn: "<i class=\"fa-brands fa-whatsapp\"></i> व्हॉट्सॲपवर शेअर करा",
            chatTitle: "<i class=\"fa-solid fa-robot\"></i> बळीराजा AI चॅट मदतनीस",
            chatPlaceholder: "तुमचा प्रश्न येथे टाका...",
            navHome: "होम",
            navCamera: "कॅमेरा",
            navDiagnosis: "अहवाल",
            navWhatsapp: "व्हॉट्सॲप"
        },
        Tamil: {
            logoSub: "பிளாண்டிக்ஸ் பாணி பயிர் நோய் கண்டறிதல் & விவசாய ஆலோசனை",
            selectCropLabel: "<i class=\"fa-solid fa-wheat-field\"></i> பயிரைத் தேர்ந்தெடுக்கவும்:",
            cropAll: "<i class=\"fa-solid fa-seedling\"></i> அனைத்து பயிர்களும்",
            cropApple: "🍎 ஆப்பிள்",
            cropPotato: "🥔 உருளைக்கிழங்கு",
            cropTomato: "🍅 தக்காளி",
            cropGrape: "🍇 திராட்சை",
            cropCorn: "🌽 சோளம்",
            cropRice: "🌾 நெல்",
            cropWheat: "🌾 கோதுமை",
            sampleApple: "🍎 ஆப்பிள் ஸ்கேப்",
            samplePotato: "🥔 உருளை பிளைட்",
            sampleTomato: "🍅 தக்காளி மோல்டு",
            sampleCorn: "🌽 சோள ரஸ்ட்",
            step1Title: "இலை படம் எடுக்கவும்",
            step1Sub: "பாதிக்கப்பட்ட இலையின் படத்தை எடுக்கவும்",
            step2Title: "AI ஹீட்மேப் சோதனை",
            step2Sub: "Grad-CAM நோய் பகுதியைக் கண்டறியும்",
            step3Title: "3-படி சிகிச்சை",
            step3Sub: "ரசாயன, இயற்கை மற்றும் குரல் ஆலோசனை",
            plantixTitle: "பிளாண்டிக்ஸ் AI பயிர் டாக்டர்",
            plantixSub: "பயிர் நோய் கண்டறிதல் மற்றும் உடனடி சிகிச்சை",
            cameraBtnText: "இலையை படம் எடுக்கவும்",
            cameraBtnSub: "மொபைல் கேமரா தானாகவே திறக்கும்",
            galleryBtnText: "<i class=\"fa-solid fa-images\"></i> கேலரியிலிருந்து தேர்வு செய்க",
            samplesTitle: "<i class=\"fa-solid fa-vial\"></i> அல்லது 1-கிளிக் சோதனை மாதிரிகள்:",
            analyzeBtn: "<i class=\"fa-solid fa-stethoscope\"></i> AI நோய் பரிசோதனை செய்",
            diagTitle: "<i class=\"fa-solid fa-square-poll-vertical\"></i> பயிர் நோய் அறிக்கை",
            tabOverview: "<i class=\"fa-solid fa-file-medical\"></i> கண்ணோட்டம்",
            tabChemical: "<i class=\"fa-solid fa-vial-circle-check\"></i> ரசாயன கட்டுப்பாடு",
            tabBiological: "<i class=\"fa-solid fa-seedling\"></i> இயற்கை சிகிச்சை",
            tabPreventative: "<i class=\"fa-solid fa-shield-halved\"></i> தடுப்பு முறைகள்",
            dosageTitle: "மருந்து அளவு கணக்கீட்டாளர்",
            voiceBtn: "<i class=\"fa-solid fa-volume-high\"></i> கேட்கவும்",
            whatsappBtn: "<i class=\"fa-brands fa-whatsapp\"></i> வாட்ஸ்அப்பில் பகிரவும்",
            chatTitle: "<i class=\"fa-solid fa-robot\"></i> கிசான் AI சாட் உதவியாளர்",
            chatPlaceholder: "உங்கள் கேள்வியை உள்ளிடவும்...",
            navHome: "முகப்பு",
            navCamera: "கேமரா",
            navDiagnosis: "அறிக்கை",
            navWhatsapp: "வாட்ஸ்அப்"
        },
        Gujarati: {
            logoSub: "પ્લાન્ટિક્સ-સ્ટાઈલ પાક રોગ નિદાન અને કૃષિ માર્ગદર્શન",
            selectCropLabel: "<i class=\"fa-solid fa-wheat-field\"></i> પાક પસંદ કરો:",
            cropAll: "<i class=\"fa-solid fa-seedling\"></i> તમામ પાક",
            cropApple: "🍎 સફરજન",
            cropPotato: "🥔 બટાકા",
            cropTomato: "🍅 ટામેટા",
            cropGrape: "🍇 દ્રાક્ષ",
            cropCorn: "🌽 મકાઈ",
            cropRice: "🌾 ડાંગર/ચોખા",
            cropWheat: "🌾 ઘઉં",
            sampleApple: "🍎 સફરજન સ્કેબ",
            samplePotato: "🥔 બટાકાનો સુકારો",
            sampleTomato: "🍅 ટામેટા મોલ્ડ",
            sampleCorn: "🌽 મકાઈ ગેરુ",
            step1Title: "પાનનો ફોટો લો",
            step1Sub: "બગડેલા પાનનો સ્પષ્ટ ફોટો પાડો",
            step2Title: "AI હીટમેપ ચકાસણી",
            step2Sub: "Grad-CAM અસરગ્રસ્ત ભાગ શોધશે",
            step3Title: "3-તબક્કાવાર સારવાર",
            step3Sub: "જૈવિક, રાસાયણિક અને વૉઇસ સલાહ",
            plantixTitle: "પ્લાન્ટિક્સ AI ક્રોપ ડૉક્ટર",
            plantixSub: "છોડના રોગોનું ત્વરિત નિદાન અને ચોક્કસ ઉપચાર",
            cameraBtnText: "પાનનો ફોટો પાડો",
            cameraBtnSub: "તમારો મોબાઈલ કેમેરા આપમેળે ખુલશે",
            galleryBtnText: "<i class=\"fa-solid fa-images\"></i> ગેલેરીમાંથી પસંદ કરો",
            samplesTitle: "<i class=\"fa-solid fa-vial\"></i> અથવા 1-ક્લિક ટેસ્ટ સેમ્પલ ચકાસો:",
            analyzeBtn: "<i class=\"fa-solid fa-stethoscope\"></i> AI રોગ નિદાન શરૂ કરો",
            diagTitle: "<i class=\"fa-solid fa-square-poll-vertical\"></i> પાક નિદાન અહેવાલ",
            tabOverview: "<i class=\"fa-solid fa-file-medical\"></i> વિગત",
            tabChemical: "<i class=\"fa-solid fa-vial-circle-check\"></i> રાસાયણિક નિયંત્રણ",
            tabBiological: "<i class=\"fa-solid fa-seedling\"></i> જૈવિક ઉપચાર",
            tabPreventative: "<i class=\"fa-solid fa-shield-halved\"></i> રક્ષણાત્મક પગલાં",
            dosageTitle: "દવા છંટકાવ પ્રમાણ કેલ્ક્યુલેટર",
            voiceBtn: "<i class=\"fa-solid fa-volume-high\"></i> સાંભળો",
            whatsappBtn: "<i class=\"fa-brands fa-whatsapp\"></i> વોટ્સએપ પર શેર કરો",
            chatTitle: "<i class=\"fa-solid fa-robot\"></i> ખેડૂત AI ચેટ મદદનીશ",
            chatPlaceholder: "તમારો પ્રશ્ન અહીં લખો...",
            navHome: "હોમ",
            navCamera: "કેમેરા",
            navDiagnosis: "અહેવાલ",
            navWhatsapp: "વોટ્સએપ"
        },
        English: {
            logoSub: "Plantix-Style Plant Disease Diagnosis & Agricultural Advisory",
            selectCropLabel: "<i class=\"fa-solid fa-wheat-field\"></i> Select Crop:",
            cropAll: "<i class=\"fa-solid fa-seedling\"></i> All Crops",
            cropApple: "🍎 Apple",
            cropPotato: "🥔 Potato",
            cropTomato: "🍅 Tomato",
            cropGrape: "🍇 Grape",
            cropCorn: "🌽 Corn",
            cropRice: "🌾 Rice",
            cropWheat: "🌾 Wheat",
            sampleApple: "🍎 Apple Scab",
            samplePotato: "🥔 Potato Blight",
            sampleTomato: "🍅 Tomato Mold",
            sampleCorn: "🌽 Corn Rust",
            step1Title: "Snap Infected Leaf",
            step1Sub: "Take a close photo of crop leaf",
            step2Title: "AI Heatmap Check",
            step2Sub: "Grad-CAM spots disease area",
            step3Title: "3-Pillar Treatment",
            step3Sub: "Get Chemical, Organic & Voice advice",
            plantixTitle: "Plantix AI Crop Doctor",
            plantixSub: "Diagnose plant diseases & get instant treatment protocol",
            cameraBtnText: "Take Photo of Diseased Leaf",
            cameraBtnSub: "Opens your phone HD camera automatically",
            galleryBtnText: "<i class=\"fa-solid fa-images\"></i> Choose Photo from Gallery",
            samplesTitle: "<i class=\"fa-solid fa-vial\"></i> Or Try 1-Click Quick Test Samples:",
            dropInstructions: "Drag & drop leaf photo here or click to browse",
            dropFormats: "Supports JPG, PNG, WEBP up to 10MB",
            analyzeBtn: "<i class=\"fa-solid fa-stethoscope\"></i> Run AI Crop Diagnosis",
            oodTitle: "Non-Crop Image Detected",
            oodDesc: "This image does not resemble standard agricultural crops.",
            diagTitle: "<i class=\"fa-solid fa-square-poll-vertical\"></i> Diagnostic Report",
            confLabel: "Confidence",
            tabOverview: "<i class=\"fa-solid fa-file-medical\"></i> Overview",
            tabChemical: "<i class=\"fa-solid fa-vial-circle-check\"></i> Chemical Control",
            tabBiological: "<i class=\"fa-solid fa-seedling\"></i> Biological / Organic",
            tabPreventative: "<i class=\"fa-solid fa-shield-halved\"></i> Preventative Checks",
            dosageTitle: "Field Spray Dosage Calculator",
            fieldAreaLabel: "Field Area:",
            voiceBtn: "<i class=\"fa-solid fa-volume-high\"></i> Read Out Loud",
            whatsappBtn: "<i class=\"fa-brands fa-whatsapp\"></i> Share Report on WhatsApp",
            printBtn: "<i class=\"fa-solid fa-print\"></i> Print / Save PDF",
            heatmapTitle: "<i class=\"fa-solid fa-bullseye\"></i> AI Detection Heatmap (Grad-CAM)",
            heatmapSub: "Highlights infected leaf spots analyzed by ViT",
            opacityLabel: "Heatmap Blend Opacity:",
            chatTitle: "<i class=\"fa-solid fa-robot\"></i> Kisan AI Chat Assistant",
            chatSub: "Ask questions about sprays, soil, or recovery",
            chatPlaceholder: "Type your query here...",
            sendBtn: "<i class=\"fa-solid fa-paper-plane\"></i> Send",
            navHome: "Home",
            navCamera: "Camera",
            navDiagnosis: "Diagnosis",
            navWhatsapp: "WhatsApp",
            disclaimer: "Recommendations are generated using indexed agricultural databases and AI guidance. Verify dosage locally."
        }
    };

function applyLanguageUI(selectedLang) {
        const tr = UI_TRANSLATIONS[selectedLang] || UI_TRANSLATIONS["Hindi"] || UI_TRANSLATIONS["English"];
        if (!tr) return;

        document.querySelectorAll("[data-i18n]").forEach(el => {
            const key = el.getAttribute("data-i18n");
            if (tr[key]) {
                if (el.tagName === "INPUT" || el.tagName === "TEXTAREA") {
                    el.placeholder = tr[key];
                } else {
                    el.innerHTML = tr[key];
                }
            }
        });

        const logoSub = document.querySelector(".logo p");
        if (logoSub && tr.logoSub) logoSub.innerText = tr.logoSub;

        if (chatInput && tr.chatPlaceholder) chatInput.placeholder = tr.chatPlaceholder;
        showToast(`Website auto-translated to ${selectedLang}`, "info");
    }

    // Language Selector Change Event
    langSelect.addEventListener("change", async () => {
        const selectedLang = langSelect.value;
        applyLanguageUI(selectedLang);

        if (currentDiagnosis) {
            setLoader(true, `Updating diagnostic advisory to ${selectedLang}...`);
            await fetchAdvice();
            setLoader(false);
            showToast(`Diagnosis updated in ${selectedLang}!`, "success");
        }
    });

    // Crop Selection Pills & Plant Search Logic
    let selectedCrop = "all";
    const cropSearchInput = document.getElementById("crop-search-input");
    const cropFilterBadge = document.getElementById("crop-filter-badge");
    const cropPills = document.querySelectorAll(".crop-pill");

    function updateCropBadge(cropText) {
        if (!cropFilterBadge) return;
        if (!selectedCrop || selectedCrop === "all") {
            cropFilterBadge.innerHTML = `<i class="fa-solid fa-bullseye"></i> Mode: <strong>All Plants (Global Mode)</strong>`;
        } else {
            cropFilterBadge.innerHTML = `<i class="fa-solid fa-bullseye"></i> Target Filter: <strong>${cropText} (Crop-Specific Precision Mode Active)</strong>`;
        }
    }

    if (cropSearchInput) {
        cropSearchInput.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase().trim();
            cropPills.forEach(pill => {
                const text = pill.innerText.toLowerCase();
                const cropVal = pill.getAttribute("data-crop") || "";
                if (cropVal === "all" || text.includes(query) || cropVal.includes(query)) {
                    pill.style.display = "inline-flex";
                } else {
                    pill.style.display = "none";
                }
            });
        });
    }

    cropPills.forEach(pill => {
        pill.addEventListener("click", () => {
            cropPills.forEach(p => p.classList.remove("active"));
            pill.classList.add("active");
            selectedCrop = pill.getAttribute("data-crop") || "all";
            const pillText = pill.innerText.trim();
            updateCropBadge(pillText);

            // Filter sample chips
            sampleBtns.forEach(sample => {
                const sampleCrop = sample.getAttribute("data-crop");
                if (selectedCrop === "all" || sampleCrop === selectedCrop) {
                    sample.style.display = "inline-flex";
                    sample.classList.add("pulse-highlight");
                    setTimeout(() => sample.classList.remove("pulse-highlight"), 1000);
                } else {
                    sample.style.display = "none";
                }
            });

            showToast(`Crop filter active: ${pillText}`, "success");
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
                🧪 Chemical: <strong>${copperGrams}g Copper Oxychloride in ${waterLiters}L Water</strong><br>
                🌿 Organic: <strong>${neemMl}ml Neem Oil Spray</strong>
            `;
        });
    }

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

    // PDF Export Report Handler
    const printBtn = document.getElementById("print-btn");
    if (printBtn) {
        printBtn.addEventListener("click", () => {
            window.print();
        });
    }

    // Share Diagnosis Report Handler (WhatsApp)
    const whatsappShareBtn = document.getElementById("whatsapp-share-btn");
    if (whatsappShareBtn) {
        whatsappShareBtn.addEventListener("click", () => {
            const diseaseEl = document.getElementById("disease-name");
            const confEl = document.getElementById("confidence-percentage");
            const diseaseTitle = diseaseEl ? diseaseEl.textContent.trim() : "Crop Diagnosis";
            const confidence = confEl ? confEl.textContent.trim() : "";
            const shareText = `🌾 *SmartFarming AI Diagnosis Report*\n\n📌 *Disease / Condition:* ${diseaseTitle}\n📊 *Confidence:* ${confidence}\n\nGenerated via SmartFarming AI Crop Doctor.`;
            const encodedText = encodeURIComponent(shareText);
            
            if (navigator.share) {
                navigator.share({
                    title: 'SmartFarming AI Diagnosis Report',
                    text: shareText
                }).catch(() => {
                    window.open(`https://api.whatsapp.com/send?text=${encodedText}`, '_blank');
                });
            } else {
                window.open(`https://api.whatsapp.com/send?text=${encodedText}`, '_blank');
            }
        });
    }

    // Mobile Bottom Navigation Bar Handlers
    const navHome = document.getElementById("nav-home");
    const navCamera = document.getElementById("nav-camera");
    const navDiagnosis = document.getElementById("nav-diagnosis");
    const navWhatsapp = document.getElementById("nav-whatsapp");
    const navItems = [navHome, navCamera, navDiagnosis, navWhatsapp];

    function setActiveNav(target) {
        navItems.forEach(item => item && item.classList.remove("active"));
        if (target) target.classList.add("active");
    }

    if (navHome) {
        navHome.addEventListener("click", () => {
            setActiveNav(navHome);
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    }

    if (navCamera) {
        navCamera.addEventListener("click", () => {
            setActiveNav(navCamera);
            if (openCameraBtn) openCameraBtn.click();
        });
    }

    if (navDiagnosis) {
        navDiagnosis.addEventListener("click", () => {
            setActiveNav(navDiagnosis);
            if (diagnosisCard && !diagnosisCard.classList.contains("hidden")) {
                diagnosisCard.scrollIntoView({ behavior: "smooth", block: "start" });
            } else {
                showToast("Please snap or upload a crop leaf photo first.", "info");
            }
        });
    }

    if (navWhatsapp) {
        navWhatsapp.addEventListener("click", () => {
            setActiveNav(navWhatsapp);
            if (whatsappModal) whatsappModal.classList.remove("hidden");
        });
    }

    // Helper: Formatter for Disease Name
    function formatDiseaseName(rawName) {
        if (!rawName) return "";
        let formatted = rawName.replace("___", ": ").replace(/_/g, " ");
        return formatted;
    }

    // ==========================================
    // 6. VOICE ASSISTANT & READ OUT LOUD MODULE
    // ==========================================
    const readVoiceBtn = document.getElementById("read-voice-btn");
    const micBtn = document.getElementById("mic-btn");
    let synth = window.speechSynthesis;
    let currentUtterance = null;
    let activeVoiceButton = null;

    const LANG_CODE_MAP = {
        Hindi: "hi-IN",
        Bengali: "bn-IN",
        Telugu: "te-IN",
        Marathi: "mr-IN",
        Tamil: "ta-IN",
        Gujarati: "gu-IN",
        English: "en-US"
    };

    const STOP_TEXT_MAP = {
        Hindi: '<i class="fa-solid fa-square-stop"></i> सुनना रोकें',
        Bengali: '<i class="fa-solid fa-square-stop"></i> থামান',
        Telugu: '<i class="fa-solid fa-square-stop"></i> ఆపండి',
        Marathi: '<i class="fa-solid fa-square-stop"></i> थांबवा',
        Tamil: '<i class="fa-solid fa-square-stop"></i> நிறுத்து',
        Gujarati: '<i class="fa-solid fa-square-stop"></i> બંધ કરો',
        English: '<i class="fa-solid fa-square-stop"></i> Stop Reading'
    };

    function stopSpeech() {
        if (synth && (synth.speaking || synth.pending)) {
            synth.cancel();
        }
        if (activeVoiceButton) {
            activeVoiceButton.classList.remove("speaking-pulse");
            if (activeVoiceButton === readVoiceBtn) {
                const selectedLang = langSelect ? langSelect.value : "English";
                const tr = UI_TRANSLATIONS[selectedLang] || UI_TRANSLATIONS["English"];
                readVoiceBtn.innerHTML = (tr && tr.voiceBtn) ? tr.voiceBtn : '<i class="fa-solid fa-volume-high"></i> Read Out Loud';
            } else {
                activeVoiceButton.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
            }
            activeVoiceButton = null;
        }
    }

    function speakText(rawText, targetBtn = readVoiceBtn) {
        if (!("speechSynthesis" in window)) {
            showToast("Speech synthesis is not supported in this browser.", "error");
            return;
        }

        // Toggle stop if already speaking with this button
        if (synth.speaking && activeVoiceButton === targetBtn) {
            stopSpeech();
            showToast("Voice playback stopped.", "info");
            return;
        }

        // Stop any active speech first
        stopSpeech();

        // Clean HTML tags and formatting from text
        let cleanText = rawText.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();

        if (!cleanText || cleanText.includes("No diagnosis analyzed yet")) {
            showToast("Please analyze a plant leaf photo first to generate treatment advice to read out loud.", "info");
            return;
        }

        const selectedLang = langSelect ? langSelect.value : "English";
        const langCode = LANG_CODE_MAP[selectedLang] || "en-US";

        currentUtterance = new SpeechSynthesisUtterance(cleanText);
        currentUtterance.lang = langCode;
        currentUtterance.rate = 0.92; // Slightly slower for clear rural accent delivery
        currentUtterance.pitch = 1.0;

        // Try selecting matching accent/language voice from browser system voices
        const availableVoices = synth.getVoices();
        if (availableVoices && availableVoices.length > 0) {
            const matchedVoice = availableVoices.find(v => 
                v.lang.toLowerCase().startsWith(langCode.split('-')[0].toLowerCase()) || 
                v.lang.toLowerCase() === langCode.toLowerCase()
            );
            if (matchedVoice) {
                currentUtterance.voice = matchedVoice;
            }
        }

        activeVoiceButton = targetBtn;

        currentUtterance.onstart = () => {
            if (activeVoiceButton) {
                activeVoiceButton.classList.add("speaking-pulse");
                if (activeVoiceButton === readVoiceBtn) {
                    const stopHtml = STOP_TEXT_MAP[selectedLang] || STOP_TEXT_MAP["English"];
                    activeVoiceButton.innerHTML = stopHtml;
                } else {
                    activeVoiceButton.innerHTML = '<i class="fa-solid fa-volume-xmark"></i>';
                }
            }
            showToast(`Reading advice out loud in ${selectedLang}...`, "info");
        };

        currentUtterance.onend = () => {
            stopSpeech();
        };

        currentUtterance.onerror = (e) => {
            console.error("Speech Synthesis Error:", e);
            stopSpeech();
            showToast("Speech output encountered an error.", "error");
        };

        synth.speak(currentUtterance);
    }

    if (readVoiceBtn) {
        readVoiceBtn.addEventListener("click", () => {
            // Find active tab content or overall treatment description
            const activeTabPanel = document.querySelector(".tab-panel.active");
            let textToSpeak = "";

            if (diseaseName && diseaseName.innerText) {
                textToSpeak += `Crop Diagnosis: ${diseaseName.innerText}. `;
            }

            if (activeTabPanel && activeTabPanel.innerText) {
                const activeTabBtn = document.querySelector(".tab-btn.active");
                const tabTitle = activeTabBtn ? activeTabBtn.innerText : "Treatment";
                textToSpeak += `${tabTitle}: ${activeTabPanel.innerText}`;
            } else {
                const overviewEl = document.getElementById("treatment-overview");
                textToSpeak += overviewEl ? overviewEl.innerText : "";
            }

            speakText(textToSpeak, readVoiceBtn);
        });
    }

    // Ensure system voices are loaded (necessary for Chrome/Edge)
    if (synth && typeof synth.onvoiceschanged !== "undefined") {
        synth.onvoiceschanged = () => {
            // Voices refreshed in background
        };
    }

    // Microphone Voice Input (Speech Recognition)
    if (micBtn) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            const recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;

            micBtn.addEventListener("click", () => {
                const selectedLang = langSelect ? langSelect.value : "English";
                recognition.lang = LANG_CODE_MAP[selectedLang] || "en-US";
                
                try {
                    recognition.start();
                    micBtn.classList.add("listening-mic");
                    showToast(`Listening... Speak your question in ${selectedLang}`, "info");
                } catch (err) {
                    recognition.stop();
                    micBtn.classList.remove("listening-mic");
                }
            });

            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                if (chatInput) {
                    chatInput.value = transcript;
                    showToast(`Captured: "${transcript}"`, "success");
                }
                micBtn.classList.remove("listening-mic");
            };

            recognition.onerror = (event) => {
                console.error("Speech Recognition error:", event.error);
                micBtn.classList.remove("listening-mic");
                showToast("Voice input error or permission denied.", "error");
            };

            recognition.onend = () => {
                micBtn.classList.remove("listening-mic");
            };
        } else {
            micBtn.style.display = "none"; // Hide if speech recognition not supported in browser
        }
    }

    // ==========================================
    // 7. CHAT ASSISTANT & INTERACTION HANDLERS
    // ==========================================
    if (chatToggle && chatDrawer) {
        chatToggle.addEventListener("click", () => {
            chatDrawer.classList.toggle("collapsed");
            chatBadge.classList.add("hidden");
        });
    }

    function appendChatMessage(sender, text) {
        if (!chatMessages) return;
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender}`;

        const textSpan = document.createElement("span");
        textSpan.innerHTML = text.replace(/\n/g, "<br>");
        msgDiv.appendChild(textSpan);

        if (sender === "bot") {
            const voiceBtn = document.createElement("button");
            voiceBtn.className = "chat-voice-btn";
            voiceBtn.setAttribute("title", "Read message out loud");
            voiceBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
            voiceBtn.addEventListener("click", () => {
                speakText(text, voiceBtn);
            });
            msgDiv.appendChild(voiceBtn);
        }

        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function handleSendChatMessage() {
        if (!chatInput) return;
        const text = chatInput.value.trim();
        if (!text) return;

        appendChatMessage("user", text);
        chatInput.value = "";

        if (!currentDiagnosis) {
            appendChatMessage("bot", "Please run a crop leaf diagnosis first so I can assist you with precise recommendations.");
            return;
        }

        appendChatMessage("bot", '<i class="fa-solid fa-spinner fa-spin"></i> Consulting Kisan AI Agronomist...');
        const tempBotMsg = chatMessages.lastElementChild;

        try {
            const response = await fetch(`${API_BASE}/api/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    disease: currentDiagnosis.disease,
                    history: chatHistory,
                    message: text,
                    language: langSelect ? langSelect.value : "English"
                })
            });

            if (!response.ok) throw new Error("Server error during chat query");

            const data = await safeJson(response);
            if (tempBotMsg) tempBotMsg.remove();

            appendChatMessage("bot", data.reply);
            chatHistory.push({ sender: "user", text: text });
            chatHistory.push({ sender: "bot", text: data.reply });

        } catch (err) {
            console.error("Chat error:", err);
            if (tempBotMsg) tempBotMsg.remove();
            appendChatMessage("bot", "Sorry, I could not fetch a response at this moment. Please check your server connection.");
        }
    }

    if (chatSendBtn) {
        chatSendBtn.addEventListener("click", handleSendChatMessage);
    }

    if (chatInput) {
        chatInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") handleSendChatMessage();
        });
    }

    if (clearChat) {
        clearChat.addEventListener("click", () => {
            chatHistory = [];
            if (chatMessages) {
                chatMessages.innerHTML = `
                    <div class="message bot">
                        Hello! I'm your AI farming advisor. Once you run a crop diagnosis, you can ask me questions about organic spraying, dosage, recovery time, or crop safety here.
                    </div>
                `;
            }
            showToast("Chat conversation cleared.", "info");
        });
    }

    // ==========================================
    // COMMUNITY CONTRIBUTION & RETRAINING MODAL
    // ==========================================
    const openContribBtn = document.getElementById("open-contribute-modal-btn");
    const closeContribBtn = document.getElementById("close-contribute-modal-btn");
    const contribModal = document.getElementById("contribute-modal");
    const contribForm = document.getElementById("contribute-form");
    const contribLabelSelect = document.getElementById("contrib-label-select");

    // Populate disease categories into select dropdown
    const knownDiseases = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
        "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
        "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
        "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
        "Strawberry___Leaf_scorch", "Strawberry___healthy",
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
    ];

    if (contribLabelSelect) {
        knownDiseases.forEach(d => {
            const opt = document.createElement("option");
            opt.value = d;
            opt.textContent = d.replace(/___/g, " - ").replace(/_/g, " ");
            contribLabelSelect.appendChild(opt);
        });
    }

    if (openContribBtn && contribModal) {
        openContribBtn.addEventListener("click", () => {
            contribModal.classList.remove("hidden");
        });
    }

    if (closeContribBtn && contribModal) {
        closeContribBtn.addEventListener("click", () => {
            contribModal.classList.add("hidden");
        });
    }

    if (contribForm) {
        contribForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById("contrib-file-input");
            const labelSelect = document.getElementById("contrib-label-select");
            const nameInput = document.getElementById("contrib-name-input");
            const cropInput = document.getElementById("contrib-crop-input");
            const notesInput = document.getElementById("contrib-notes-input");

            if (!fileInput.files || !fileInput.files[0]) {
                showToast("Please choose an image file to upload.", "error");
                return;
            }

            const formData = new FormData();
            formData.append("file", fileInput.files[0]);
            formData.append("label", labelSelect.value);
            if (nameInput.value) formData.append("contributor_name", nameInput.value);
            if (cropInput.value) formData.append("crop_type", cropInput.value);
            if (notesInput.value) formData.append("notes", notesInput.value);

            try {
                setLoader(true, "Uploading community training sample to database...");
                const response = await fetch(`${API_BASE}/api/contribute`, {
                    method: "POST",
                    body: formData
                });
                setLoader(false);

                if (!response.ok) {
                    const errData = await response.json().catch(() => ({}));
                    throw new Error(errData.detail || "Upload failed");
                }

                const res = await response.json();
                showToast("🌱 " + res.message, "success");
                contribForm.reset();
                contribModal.classList.add("hidden");
            } catch (err) {
                setLoader(false);
                showToast("Contribution Error: " + err.message, "error");
            }
        });
    }

    // ============================================================
    // THREE.JS 3D BOTANICAL DIGITAL TWIN INTERACTIVE ENGINE
    // ============================================================
    function init3DPlantHero() {
        const canvas = document.getElementById("three-plant-canvas");
        if (!canvas || typeof THREE === "undefined") return;

        const container = canvas.parentElement;
        const width = container.clientWidth || 500;
        const height = container.clientHeight || 380;

        // Scene, Camera, Renderer
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x03080c, 0.04);

        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
        camera.position.set(0, 0, 5.5);

        const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x10b981, 0.6);
        scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
        dirLight.position.set(5, 8, 5);
        scene.add(dirLight);

        const pointLight = new THREE.PointLight(0x34d399, 2.5, 10);
        pointLight.position.set(0, 1.5, 2);
        scene.add(pointLight);

        // Group container for easy rotation
        const plantGroup = new THREE.Group();
        scene.add(plantGroup);

        // 1. Procedural 3D Botanical Leaf Geometry
        const leafShape = new THREE.Shape();
        leafShape.moveTo(0, -1.8);
        leafShape.bezierCurveTo(1.2, -0.9, 1.4, 0.5, 0, 1.8);
        leafShape.bezierCurveTo(-1.4, 0.5, -1.2, -0.9, 0, -1.8);

        const extrudeSettings = {
            depth: 0.08,
            bevelEnabled: true,
            bevelSegments: 4,
            steps: 2,
            bevelSize: 0.04,
            bevelThickness: 0.03
        };
        const leafGeo = new THREE.ExtrudeGeometry(leafShape, extrudeSettings);

        // Materials for different display modes
        const organicMat = new THREE.MeshStandardMaterial({
            color: 0x10b981,
            roughness: 0.25,
            metalness: 0.1,
            emissive: 0x064e3b,
            emissiveIntensity: 0.35,
            wireframe: false
        });

        const neuralMat = new THREE.MeshStandardMaterial({
            color: 0x34d399,
            wireframe: true,
            emissive: 0x10b981,
            emissiveIntensity: 0.8
        });

        const leafMesh = new THREE.Mesh(leafGeo, organicMat);
        leafMesh.rotation.x = -0.15;
        plantGroup.add(leafMesh);

        // 2. Central Leaf Vein
        const veinGeo = new THREE.CylinderGeometry(0.04, 0.015, 3.4, 12);
        const veinMat = new THREE.MeshBasicMaterial({ color: 0xa7f3d0 });
        const vein = new THREE.Mesh(veinGeo, veinMat);
        vein.position.set(0, 0, 0.06);
        plantGroup.add(vein);

        // 3. Floating Chloroplast Bio-Particles
        const particleCount = 180;
        const particleGeo = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const scales = new Float32Array(particleCount);

        for (let i = 0; i < particleCount * 3; i += 3) {
            positions[i] = (Math.random() - 0.5) * 4.2;
            positions[i + 1] = (Math.random() - 0.5) * 4.2;
            positions[i + 2] = (Math.random() - 0.5) * 3.5;
            scales[i / 3] = Math.random();
        }

        particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const particleMat = new THREE.PointsMaterial({
            color: 0x34d399,
            size: 0.08,
            transparent: true,
            opacity: 0.85,
            blending: THREE.AdditiveBlending
        });

        const particles = new THREE.Points(particleGeo, particleMat);
        plantGroup.add(particles);

        // 4. Interactive Drag-to-Rotate & Inertia
        let isDragging = false;
        let prevMousePos = { x: 0, y: 0 };
        let rotationVelocity = { x: 0.003, y: 0.007 };

        function onPointerDown(e) {
            isDragging = true;
            prevMousePos = { x: e.clientX || e.touches[0].clientX, y: e.clientY || e.touches[0].clientY };
        }

        function onPointerMove(e) {
            const clientX = e.clientX || (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
            const clientY = e.clientY || (e.touches && e.touches[0] ? e.touches[0].clientY : 0);

            if (isDragging) {
                const deltaX = clientX - prevMousePos.x;
                const deltaY = clientY - prevMousePos.y;

                rotationVelocity.y = deltaX * 0.005;
                rotationVelocity.x = deltaY * 0.005;

                plantGroup.rotation.y += rotationVelocity.y;
                plantGroup.rotation.x += rotationVelocity.x;

                prevMousePos = { x: clientX, y: clientY };
            }
        }

        function onPointerUp() {
            isDragging = false;
        }

        canvas.addEventListener("mousedown", onPointerDown);
        window.addEventListener("mousemove", onPointerMove);
        window.addEventListener("mouseup", onPointerUp);

        canvas.addEventListener("touchstart", onPointerDown, { passive: true });
        window.addEventListener("touchmove", onPointerMove, { passive: true });
        window.addEventListener("touchend", onPointerUp);

        // 5. Mode Switchers (Organic / ViT Attention Grid / Bio-Particles)
        const organicBtn = document.getElementById("mode-organic-btn");
        const neuralBtn = document.getElementById("mode-neural-btn");
        const particlesBtn = document.getElementById("mode-particles-btn");
        const modeBtns = [organicBtn, neuralBtn, particlesBtn];

        function setMode(mode) {
            modeBtns.forEach(btn => btn && btn.classList.remove("active"));
            if (mode === "organic") {
                if (organicBtn) organicBtn.classList.add("active");
                leafMesh.visible = true;
                leafMesh.material = organicMat;
                vein.visible = true;
                particles.visible = true;
            } else if (mode === "neural") {
                if (neuralBtn) neuralBtn.classList.add("active");
                leafMesh.visible = true;
                leafMesh.material = neuralMat;
                vein.visible = false;
                particles.visible = true;
            } else if (mode === "particles") {
                if (particlesBtn) particlesBtn.classList.add("active");
                leafMesh.visible = false;
                vein.visible = false;
                particles.visible = true;
                particleMat.size = 0.12;
            }
        }

        if (organicBtn) organicBtn.addEventListener("click", () => setMode("organic"));
        if (neuralBtn) neuralBtn.addEventListener("click", () => setMode("neural"));
        if (particlesBtn) particlesBtn.addEventListener("click", () => setMode("particles"));

        // Resize handler
        window.addEventListener("resize", () => {
            const newW = container.clientWidth;
            const newH = container.clientHeight;
            camera.aspect = newW / newH;
            camera.updateProjectionMatrix();
            renderer.setSize(newW, newH);
        });

        // 6. Animation Render Loop (60 FPS)
        let clock = new THREE.Clock();
        function animate() {
            requestAnimationFrame(animate);

            const elapsedTime = clock.getElapsedTime();

            // Continuous smooth rotation with damping
            if (!isDragging) {
                plantGroup.rotation.y += rotationVelocity.y;
                plantGroup.rotation.x += rotationVelocity.x;

                // Gentle return to base speed
                rotationVelocity.y += (0.006 - rotationVelocity.y) * 0.05;
                rotationVelocity.x += (0.0 - rotationVelocity.x) * 0.05;
            }

            // Natural organic floating bounce
            plantGroup.position.y = Math.sin(elapsedTime * 1.5) * 0.12;

            // Animate floating bio-particles
            particles.rotation.y = elapsedTime * 0.08;
            particles.rotation.x = Math.sin(elapsedTime * 0.5) * 0.1;

            renderer.render(scene, camera);
        }
        animate();
    }

    // Initialize 3D Botanical Hero
    init3DPlantHero();
});


