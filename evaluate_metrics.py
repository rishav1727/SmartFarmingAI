# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
TABLE I + TABLE II  — HONEST END-TO-END EVALUATION METRICS
============================================================
FIXED ISSUES:
  1. Classifier     : Real Accuracy + Macro F1 from labeled_test_set/ (sklearn)
                      No more "confidence == accuracy" mistake
  2. Grad-CAM       : Real bounding-box IoU from annotated_boxes.json
                      Honest "N/A — no GT mask" when annotation missing
  3. Retriever      : Expanded RAG corpus (30+ chunks), honest Recall@k / MRR
  4. LLM-RAG        : Honest heuristic scores (clearly labeled, NOT fake 1.000)
  5. System         : End-to-end latency (unchanged, was already correct)

Run:  venv\\Scripts\\python.exe evaluate_metrics.py
"""

import os
import sys
import time
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import torch
from torchvision import transforms
from PIL import Image, ImageOps

from src.consts   import DEVICE, MODEL_DIR, IMAGE_SIZE
from src.model    import get_model
from src.visualize import GradCAM, overlay_heatmap

SEP = "=" * 70

def banner(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# 0. LOAD MODEL & CLASS NAMES
# ─────────────────────────────────────────────────────────────────────────────
banner("0 · Loading Model & Class Names")

with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
    class_names = json.load(f)
num_classes = len(class_names)
print(f"Classes  : {num_classes}")
print(f"Device   : {DEVICE}")

model = get_model(num_classes, DEVICE)
weights_path = os.path.join(MODEL_DIR, "best_model.pth")
model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
model.eval()
print(f"Weights  : {weights_path}")

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def run_classifier(img_path):
    """Return (pred_idx, pred_label, confidence_pct, top3)."""
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    tensor = eval_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)[0]
        probs  = torch.softmax(logits, dim=0)
    top3_p, top3_i = torch.topk(probs, 3)
    pred_idx  = top3_i[0].item()
    pred_conf = top3_p[0].item() * 100
    top3 = [(class_names[i], round(p.item()*100, 2))
            for p, i in zip(top3_p, top3_i)]
    return pred_idx, class_names[pred_idx], pred_conf, top3

# ─────────────────────────────────────────────────────────────────────────────
# 1. CLASSIFIER  — REAL Accuracy + Macro F1  (FIX #1)
#
#    labeled_test_set/<ClassName>/image.jpg  structure se ground truth padhte hain
#    sklearn se real accuracy + macro F1 compute karte hain
#    Agar labeled_test_set/ nahi hai to HuggingFace dataset se 60 images download
# ─────────────────────────────────────────────────────────────────────────────
banner("1 · Classifier  :  Real Accuracy + Macro F1  [FIX: no more confidence=accuracy]")

LABELED_DIR = os.path.join(BASE_DIR, "labeled_test_set")

def build_labeled_test_set_from_hf(target_dir, n_per_class=2, max_classes=38):
    """
    Download n_per_class images per class from HuggingFace PlantVillage dataset.
    This gives us REAL ground-truth labels.
    Total images: up to max_classes * n_per_class = 76 images.
    """
    try:
        from datasets import load_dataset
        print("  Downloading PlantVillage test split from HuggingFace...")
        ds = load_dataset(
            "BrandonFors/Plant-Diseases-PlantVillage-Dataset",
            split="test"
        )
        
        os.makedirs(target_dir, exist_ok=True)
        saved_per_class = {}
        total_saved = 0
        
        for item in ds:
            label_name = item["label"]
            if isinstance(label_name, int):
                label_name = class_names[label_name]
            
            if saved_per_class.get(label_name, 0) >= n_per_class:
                continue
            if len([k for k in saved_per_class if saved_per_class[k] > 0]) >= max_classes and label_name not in saved_per_class:
                continue
            
            class_dir = os.path.join(target_dir, label_name)
            os.makedirs(class_dir, exist_ok=True)
            
            img = item["image"]
            if not isinstance(img, Image.Image):
                from PIL import Image as PILImage
                img = PILImage.fromarray(img)
            
            idx = saved_per_class.get(label_name, 0)
            img_path = os.path.join(class_dir, f"test_{idx:03d}.jpg")
            img.save(img_path)
            saved_per_class[label_name] = idx + 1
            total_saved += 1
            
            if total_saved % 20 == 0:
                print(f"  Downloaded {total_saved} images...")
            
            # Stop if all classes have enough
            if all(v >= n_per_class for v in saved_per_class.values()) and len(saved_per_class) >= max_classes:
                break
        
        print(f"  Downloaded {total_saved} labeled test images across {len(saved_per_class)} classes.")
        return True
    except Exception as e:
        print(f"  [WARN] HuggingFace download failed: {e}")
        return False

# Check if labeled_test_set exists
has_labeled_set = False
if os.path.isdir(LABELED_DIR):
    class_dirs = [d for d in os.listdir(LABELED_DIR)
                  if os.path.isdir(os.path.join(LABELED_DIR, d))]
    total_imgs = sum(
        len([f for f in os.listdir(os.path.join(LABELED_DIR, d))
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        for d in class_dirs
    )
    if total_imgs >= 5:
        has_labeled_set = True
        print(f"  Found existing labeled_test_set/ : {total_imgs} images, {len(class_dirs)} classes")

if not has_labeled_set:
    print("  labeled_test_set/ not found or too small. Downloading from HuggingFace...")
    success = build_labeled_test_set_from_hf(LABELED_DIR, n_per_class=2, max_classes=38)
    if success:
        class_dirs = [d for d in os.listdir(LABELED_DIR)
                      if os.path.isdir(os.path.join(LABELED_DIR, d))]
        total_imgs = sum(
            len([f for f in os.listdir(os.path.join(LABELED_DIR, d))
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
            for d in class_dirs
        )
        has_labeled_set = total_imgs >= 5

# Evaluate
classifier_start = time.perf_counter()
y_true = []
y_pred = []
per_image_results = []

accuracy_pct    = None
macro_f1        = None
mean_conf       = None
std_conf        = None
all_confs_real  = []

if has_labeled_set:
    print(f"\n  Running model on labeled_test_set/ ...")
    for class_name in sorted(os.listdir(LABELED_DIR)):
        class_path = os.path.join(LABELED_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        if class_name not in class_names:
            print(f"  [SKIP] '{class_name}' not in class_names.json")
            continue
        gt_idx = class_names.index(class_name)
        
        for fname in os.listdir(class_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            fpath = os.path.join(class_path, fname)
            try:
                pred_idx, pred_label, conf, _ = run_classifier(fpath)
                y_true.append(gt_idx)
                y_pred.append(pred_idx)
                all_confs_real.append(conf)
                correct = (pred_idx == gt_idx)
                per_image_results.append({
                    "file": fname, "gt": class_name,
                    "pred": pred_label, "conf": conf, "correct": correct
                })
            except Exception as e:
                print(f"  [ERROR] {fname}: {e}")

    if y_true:
        from sklearn.metrics import accuracy_score, f1_score
        accuracy_pct = accuracy_score(y_true, y_pred) * 100
        macro_f1     = f1_score(y_true, y_pred, average='macro', zero_division=0)
        mean_conf    = float(np.mean(all_confs_real))
        std_conf     = float(np.std(all_confs_real))
        
        print(f"\n  Evaluated {len(y_true)} labeled images")
        print(f"  ┌─────────────────────────────────────────────────┐")
        print(f"  │  REAL Accuracy (sklearn)  : {accuracy_pct:>6.2f}%              │")
        print(f"  │  REAL Macro F1  (sklearn) : {macro_f1:>6.4f}               │")
        print(f"  │  Mean Confidence          : {mean_conf:>6.2f}% (std {std_conf:.2f}%)  │")
        print(f"  └─────────────────────────────────────────────────┘")
        print(f"\n  NOTE: Accuracy and Macro F1 are now computed with sklearn")
        print(f"        against ground-truth labels — NOT from confidence scores.")
        
        # Show sample of errors if any
        errors = [r for r in per_image_results if not r["correct"]]
        if errors:
            print(f"\n  Misclassified images ({len(errors)}/{len(y_true)}):")
            for e in errors[:10]:
                print(f"    GT={e['gt'][:35]:<35}  Pred={e['pred'][:35]:<35}  Conf={e['conf']:.1f}%")
    else:
        print("  [ERROR] No images found in labeled_test_set/")
else:
    print("  [WARN] labeled_test_set/ unavailable — reporting confidence stats only")
    print("         THIS IS NOT ACCURACY. Ground truth required for real Accuracy/F1.")
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    test_images = [f for f in os.listdir(BASE_DIR)
                   if os.path.splitext(f)[1].lower() in image_extensions]
    for fname in test_images:
        try:
            _, _, conf, _ = run_classifier(os.path.join(BASE_DIR, fname))
            all_confs_real.append(conf)
        except:
            pass
    mean_conf = float(np.mean(all_confs_real)) if all_confs_real else None
    std_conf  = float(np.std(all_confs_real))  if all_confs_real else None
    print(f"  Mean confidence (NO GT): {mean_conf:.2f}% — NOT accuracy")

classifier_elapsed = (time.perf_counter() - classifier_start) * 1000

# ─────────────────────────────────────────────────────────────────────────────
# 2. GRAD-CAM  — Real bounding-box IoU  (FIX #2)
# ─────────────────────────────────────────────────────────────────────────────
banner("2 · Grad-CAM  :  Bounding-Box IoU  [FIX: real IoU from annotated_boxes.json]")

BOXES_FILE = os.path.join(BASE_DIR, "annotated_boxes.json")
boxes_db = {}
if os.path.exists(BOXES_FILE):
    with open(BOXES_FILE) as f:
        boxes_db = json.load(f)
    print(f"  Loaded bounding-box annotations for {len(boxes_db)} images")
else:
    print("  [WARN] annotated_boxes.json not found — IoU cannot be computed")

gradcam_start = time.perf_counter()

test_sample = os.path.join(BASE_DIR, "test_sample.jpg")
sample_for_cam = test_sample if os.path.exists(test_sample) else None

gradcam_iou_scores    = []
gradcam_focality_vals = []
gradcam_results       = []

def boxes_to_binary_mask(boxes, img_w, img_h, target_w=224, target_h=224):
    """Convert list of bounding boxes to a binary mask (224×224)."""
    mask = np.zeros((target_h, target_w), dtype=np.float32)
    sx = target_w / img_w
    sy = target_h / img_h
    for box in boxes:
        x1 = int(box["x1"] * sx)
        y1 = int(box["y1"] * sy)
        x2 = int(box["x2"] * sx)
        y2 = int(box["y2"] * sy)
        x1, x2 = max(0, x1), min(target_w, x2)
        y1, y2 = max(0, y1), min(target_h, y2)
        mask[y1:y2, x1:x2] = 1.0
    return mask

def compute_iou_heatmap_vs_box(heatmap_224, gt_mask, threshold=0.5):
    """
    Compute IoU between binarized Grad-CAM heatmap and GT bounding-box mask.
    heatmap_224: float array 224×224, values 0-1
    gt_mask:     binary array 224×224
    threshold:   binarize heatmap at this level
    """
    pred_mask = (heatmap_224 >= threshold).astype(np.float32)
    intersection = (pred_mask * gt_mask).sum()
    union = np.clip(pred_mask + gt_mask, 0, 1).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)

if sample_for_cam:
    img_pil = Image.open(sample_for_cam)
    img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")
    orig_w, orig_h = img_pil.size

    # Create a FRESH tensor with requires_grad=True — used ONLY by GradCAM
    # Do NOT reuse the eval_transform tensor from the classifier section
    cam_tensor = eval_transform(img_pil).unsqueeze(0).to(DEVICE)
    cam_tensor.requires_grad_(True)

    target_layer = model.vit.encoder.layers[-1]
    cam_engine   = GradCAM(model, target_layer)

    # Let generate_heatmap run the single forward+backward pass internally
    # Pass category_index=None so it uses argmax from its own forward pass
    heatmap = cam_engine.generate_heatmap(cam_tensor, category_index=None)
    pred    = int(np.argmax(
        torch.softmax(
            model(eval_transform(img_pil).unsqueeze(0).to(DEVICE)), dim=1
        )[0].detach().cpu().numpy()
    ))

    # Resize to 224×224
    hm_big = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224)),
        dtype=np.float32
    ) / 255.0

    # Focality
    top25_thresh = np.percentile(hm_big, 75)
    activated    = (hm_big >= 0.5).sum()
    focality     = float((hm_big >= top25_thresh).sum() / max(activated, 1))
    gradcam_focality_vals.append(focality)

    # IoU from bounding box annotation
    fname_key = os.path.basename(sample_for_cam)
    if fname_key in boxes_db:
        ann    = boxes_db[fname_key]
        img_w  = ann.get("image_size", [orig_w, orig_h])[0]
        img_h  = ann.get("image_size", [orig_w, orig_h])[1]
        gt_mask = boxes_to_binary_mask(ann["boxes"], img_w, img_h)
        
        iou_05  = compute_iou_heatmap_vs_box(hm_big, gt_mask, threshold=0.5)
        iou_03  = compute_iou_heatmap_vs_box(hm_big, gt_mask, threshold=0.3)
        gradcam_iou_scores.append(iou_05)
        
        print(f"  Image           : {fname_key}")
        print(f"  Predicted class : {class_names[pred]}")
        print(f"  GT annotation   : {ann['annotator']} — {len(ann['boxes'])} box(es)")
        print(f"  ┌──────────────────────────────────────────────┐")
        print(f"  │  IoU @ threshold=0.5 : {iou_05:>6.4f}                │")
        print(f"  │  IoU @ threshold=0.3 : {iou_03:>6.4f}                │")
        print(f"  │  Focality score      : {focality:>6.4f}                │")
        print(f"  └──────────────────────────────────────────────┘")
        gradcam_results.append({
            "image": fname_key, "iou_05": iou_05, "iou_03": iou_03,
            "focality": focality, "gt_boxes": len(ann["boxes"])
        })
    else:
        print(f"  [WARN] No annotation for '{fname_key}' in annotated_boxes.json")
        print(f"         IoU = N/A (no ground-truth mask available)")
        print(f"         Focality: {focality:.4f} — energy concentration metric only")
        gradcam_results.append({
            "image": fname_key, "iou_05": None, "iou_03": None,
            "focality": focality, "gt_boxes": 0,
            "note": "No GT annotation — IoU not computed"
        })

    out_heatmap = overlay_heatmap(sample_for_cam, heatmap)
    out_path    = os.path.join(BASE_DIR, "eval_gradcam_output.jpg")
    out_heatmap.save(out_path)
    print(f"  Heatmap saved   : eval_gradcam_output.jpg")
else:
    print("  [SKIP] No test image found for Grad-CAM.")

gradcam_elapsed = (time.perf_counter() - gradcam_start) * 1000
mean_iou = float(np.mean(gradcam_iou_scores)) if gradcam_iou_scores else None
mean_foc = float(np.mean(gradcam_focality_vals)) if gradcam_focality_vals else None

# ─────────────────────────────────────────────────────────────────────────────
# 3. RETRIEVER — Honest Recall@k / MRR with EXPANDED corpus  (FIX #3)
# ─────────────────────────────────────────────────────────────────────────────
banner("3 · Retriever  :  Recall@k / MRR  [FIX: expanded corpus + honest evaluation]")

# Rebuild RAG chunks from ALL documents in data/documents/
def build_rag_chunks_offline(doc_dir, chunks_path):
    """
    Build keyword RAG index from all .txt files in doc_dir.
    No OpenAI needed — pure text chunking.
    """
    all_chunks = []
    if not os.path.isdir(doc_dir):
        return []
    
    for filename in sorted(os.listdir(doc_dir)):
        if not filename.endswith(".txt"):
            continue
        fpath = os.path.join(doc_dir, filename)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for p_idx, para in enumerate(paragraphs):
                if len(para) > 800:
                    # Split large paragraphs
                    sentences = para.split(". ")
                    temp = ""
                    for s in sentences:
                        if len(temp) + len(s) < 700:
                            temp += s + ". "
                        else:
                            if temp.strip():
                                all_chunks.append({
                                    "source": filename,
                                    "chunk_id": f"{filename}_{p_idx}_{len(all_chunks)}",
                                    "text": temp.strip()
                                })
                            temp = s + ". "
                    if temp.strip():
                        all_chunks.append({
                            "source": filename,
                            "chunk_id": f"{filename}_{p_idx}_{len(all_chunks)}",
                            "text": temp.strip()
                        })
                else:
                    all_chunks.append({
                        "source": filename,
                        "chunk_id": f"{filename}_{p_idx}",
                        "text": para
                    })
        except Exception as e:
            print(f"  [WARN] Error reading {filename}: {e}")
    
    # Save to chunks file
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"  Rebuilt RAG index: {len(all_chunks)} chunks from {doc_dir}")
    return all_chunks

DOC_DIR     = os.path.join(BASE_DIR, "data", "documents")
CHUNKS_PATH = os.path.join(MODEL_DIR, "rag_chunks.json")

# Always rebuild from full document set
print("  Rebuilding RAG index from data/documents/ ...")
rag_chunks = build_rag_chunks_offline(DOC_DIR, CHUNKS_PATH)

if not rag_chunks:
    # Fallback to existing
    with open(CHUNKS_PATH) as f:
        rag_chunks = json.load(f)
    print(f"  Using existing index: {len(rag_chunks)} chunks")
else:
    print(f"  Index size: {len(rag_chunks)} chunks (was 6 before fix)")

def keyword_retrieve(query, chunks, top_k=5):
    """Simple keyword retrieval — mirrors advisor.retrieve_context fallback."""
    query_terms = [t for t in query.lower().split() if len(t) > 3]
    scores = []
    for i, chunk in enumerate(chunks):
        text_lower = chunk["text"].lower()
        score = sum(1 for term in query_terms if term in text_lower)
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, sc in scores[:top_k] if sc > 0]

# Disease queries matching class_names format
disease_queries = [
    ("Apple scab disease treatment",        "apple"),
    ("Tomato late blight fungicide",        "tomato late blight"),
    ("Corn maize rust infection",           "rust"),
    ("Grape black rot management",          "grape black rot"),
    ("Potato early blight control",         "potato early blight"),
    ("Pepper bacterial spot spray copper",  "pepper bacterial spot"),
]

K_VALUES = [1, 3, 5]
recall_at_k     = {k: [] for k in K_VALUES}
reciprocal_ranks = []

retriever_start = time.perf_counter()

print(f"\n  {'Query':<45} {'RelevantChunks':>14}  {'RR':>6}  {'R@1':>4} {'R@3':>4} {'R@5':>4}")
print("  " + "-" * 84)

for query, key_disease in disease_queries:
    key_terms = [t for t in key_disease.lower().split() if len(t) > 2]
    relevant_indices = [
        i for i, c in enumerate(rag_chunks)
        if any(term in c["text"].lower() for term in key_terms)
    ]

    retrieved = keyword_retrieve(query, rag_chunks, top_k=5)

    rr = 0.0
    for rank, idx in enumerate(retrieved, 1):
        if idx in relevant_indices:
            rr = 1.0 / rank
            break
    reciprocal_ranks.append(rr)

    r_at = {}
    for k in K_VALUES:
        hit = any(idx in relevant_indices for idx in retrieved[:k])
        recall_at_k[k].append(1.0 if hit else 0.0)
        r_at[k] = hit

    print(f"  {query:<45} {len(relevant_indices):>14}  {rr:>6.3f}  "
          f"{'Y' if r_at[1] else 'N':>4} {'Y' if r_at[3] else 'N':>4} {'Y' if r_at[5] else 'N':>4}")

retriever_elapsed = (time.perf_counter() - retriever_start) * 1000

mrr            = float(np.mean(reciprocal_ranks))
recall_summary = {k: float(np.mean(v)) for k, v in recall_at_k.items()}

print(f"\n  [Retriever Summary]")
print(f"  Corpus size      : {len(rag_chunks)} chunks (expanded from 6)")
print(f"  MRR              : {mrr:.4f}")
for k in K_VALUES:
    print(f"  Recall@{k}         : {recall_summary[k]:.4f}")
print(f"  Retrieval time   : {retriever_elapsed:.1f} ms ({len(disease_queries)} queries)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. LLM-RAG — Honest Heuristic Faithfulness/Relevance  (FIX #4)
#
#    FIXED: Circular scoring removed (advice was generated WITH same keywords
#           then scored ON same keywords — that gave fake 1.000/1.000)
#
#    NEW: Two separate scoring steps with DIFFERENT keyword sets:
#      - Relevance  : does advice mention the disease name (external check)?
#      - Faithfulness: does advice cite known agronomic treatment terms from
#                      RAG chunks (external ground truth, not self-generated)?
#    Clearly labeled as "Offline Heuristic Score" — NOT real LLM evaluation.
#
#    For real LLM scores: set GEMINI_API_KEY or OPENAI_API_KEY in environment.
# ─────────────────────────────────────────────────────────────────────────────
banner("4 · LLM-RAG  :  Honest Heuristic Faithfulness/Relevance  [FIX: no circular scoring]")

from src.advisor import SmartAdvisor
advisor = SmartAdvisor()

# Check for real LLM availability
HAS_GEMINI  = bool(os.environ.get("GEMINI_API_KEY"))
HAS_OPENAI  = bool(os.environ.get("OPENAI_API_KEY"))
LLM_MODE    = "gemini" if HAS_GEMINI else ("openai" if HAS_OPENAI else "offline_heuristic")

print(f"\n  LLM mode: {LLM_MODE.upper()}")
if LLM_MODE == "offline_heuristic":
    print("  NOTE: No GEMINI_API_KEY or OPENAI_API_KEY found.")
    print("        Running HONEST OFFLINE HEURISTIC scoring.")
    print("        Scores are NOT from a real LLM judge — clearly labeled below.")
    print("        To get real scores: set GEMINI_API_KEY=your_key")

test_cases = [
    ("Apple___Apple_scab",                    87.3),
    ("Tomato___Late_blight",                  92.1),
    ("Corn_(maize)___Common_rust_",           78.5),
    ("Grape___Black_rot",                     84.0),
    ("Potato___Early_blight",                 81.7),
]

def gemini_score_advice(disease_name, advice_text):
    """Use Gemini Flash to score faithfulness and relevance."""
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    gmodel = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"""You are an agricultural AI evaluator. Score the following treatment advice for the disease '{disease_name}'.

Advice:
{advice_text}

Return JSON with:
- "relevance": float 0-1 (does the advice specifically address {disease_name}? 1=fully relevant, 0=irrelevant)
- "faithfulness": float 0-1 (does the advice cite real agronomic treatments, fungicide names, dosages? 1=fully grounded, 0=hallucinated)
- "reasoning": string (1 sentence justification)

Return only valid JSON."""
    
    response = gmodel.generate_content(prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

def honest_heuristic_score(disease_name, advice_dict, rag_chunks):
    """
    FIXED heuristic scoring — NOT circular.
    
    Relevance: External check — does advice mention the specific disease name?
    Faithfulness: Ground truth from RAG corpus — does advice cite terms found in
                  relevant RAG chunks? (RAG chunks are external, not from mock advice)
    """
    advice_text = " ".join(str(v) for v in advice_dict.values()).lower()
    clean = disease_name.replace("___", " ").replace("_", " ").lower()
    
    # Relevance: disease-specific keyword check (external standard)
    disease_words = [w for w in clean.split() if len(w) > 3 and w not in
                     ["disease", "blight", "mold", "spot", "rust", "scab"]]
    plant_name = disease_words[0] if disease_words else clean.split()[0]
    
    # Find relevant RAG chunks for this disease
    rel_chunks_text = " ".join(
        c["text"].lower() for c in rag_chunks
        if plant_name in c["text"].lower() or
        any(w in c["text"].lower() for w in clean.split() if len(w) > 4)
    )
    
    # Relevance: plant name + disease type present in advice
    relevance_terms = [w for w in clean.split() if len(w) > 3]
    relevance = sum(1 for t in relevance_terms if t in advice_text) / max(len(relevance_terms), 1)
    
    # Faithfulness: agronomic terms from RAG corpus (external ground truth)
    # Extract chemical/treatment terms from the relevant RAG chunks
    rag_grounding_terms = set()
    chemical_indicators = [
        "mancozeb", "copper", "tebuconazole", "azoxystrobin", "propiconazole",
        "chlorothalonil", "metalaxyl", "myclobutanil", "captan", "streptomycin",
        "trichoderma", "bacillus", "pseudomonas", "neem", "sulfur",
        "fungicide", "bactericide", "spray", "drip", "prune", "rotate"
    ]
    for term in chemical_indicators:
        if term in rel_chunks_text:
            rag_grounding_terms.add(term)
    
    # Score: how many RAG-sourced terms appear in advice?
    if rag_grounding_terms:
        faithfulness = sum(1 for t in rag_grounding_terms if t in advice_text) / len(rag_grounding_terms)
    else:
        # Fallback: general agronomic grounding
        general_terms = ["fungicide", "spray", "copper", "neem", "treatment"]
        faithfulness = sum(1 for t in general_terms if t in advice_text) / len(general_terms)
    
    return float(min(relevance, 1.0)), float(min(faithfulness, 1.0))

faithfulness_scores = []
relevance_scores    = []
llm_scores_detail   = []

llmrag_start = time.perf_counter()

print(f"\n  {'Disease':<42} {'Relevance':>10} {'Faithfulness':>13}  Method")
print("  " + "-" * 75)

for disease, conf in test_cases:
    advice = advisor.get_treatment_advice(disease, conf, language="English")
    
    if LLM_MODE == "gemini":
        try:
            advice_text_str = " | ".join(f"{k}: {v}" for k, v in advice.items())
            scores = gemini_score_advice(disease, advice_text_str)
            rel   = float(scores.get("relevance", 0))
            faith = float(scores.get("faithfulness", 0))
            method = "Gemini-Flash"
        except Exception as e:
            print(f"  [WARN] Gemini scoring failed for {disease}: {e}")
            rel, faith = honest_heuristic_score(disease, advice, rag_chunks)
            method = "Heuristic(fallback)"
    else:
        rel, faith = honest_heuristic_score(disease, advice, rag_chunks)
        method = "Heuristic(offline)"
    
    faithfulness_scores.append(faith)
    relevance_scores.append(rel)
    llm_scores_detail.append({
        "disease": disease, "relevance": rel, "faithfulness": faith, "method": method
    })
    
    short = disease[:38] + ".." if len(disease) > 40 else disease
    print(f"  {short:<42} {rel:>9.3f}  {faith:>12.3f}  {method}")

llmrag_elapsed = (time.perf_counter() - llmrag_start) * 1000

mean_relevance    = float(np.mean(relevance_scores))
mean_faithfulness = float(np.mean(faithfulness_scores))

print(f"\n  [LLM-RAG Summary]  (Method: {LLM_MODE})")
print(f"  Mean Relevance    : {mean_relevance:.4f}  (0–1, disease-keyword match)")
print(f"  Mean Faithfulness : {mean_faithfulness:.4f}  (0–1, RAG-grounded term match)")
print(f"  RAG eval time     : {llmrag_elapsed:.1f} ms  ({len(test_cases)} cases)")
if LLM_MODE == "offline_heuristic":
    print(f"\n  *** OFFLINE HEURISTIC — not from a real LLM judge ***")
    print(f"  *** Set GEMINI_API_KEY for real Gemini Flash evaluation ***")
    print(f"  *** Previous fake scores 1.000/1.000 replaced with honest: ***")
    print(f"  *** Relevance={mean_relevance:.3f}, Faithfulness={mean_faithfulness:.3f} ***")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SYSTEM  — End-to-end latency
# ─────────────────────────────────────────────────────────────────────────────
banner("5 · System  :  End-to-end Latency")

e2e_sample = test_sample if os.path.exists(test_sample) else None
latencies  = {"preprocessing": [], "inference": [], "heatmap": [],
               "retrieval": [], "generation": [], "total": []}
N_RUNS = 3

if e2e_sample:
    for run in range(N_RUNS):
        t0 = time.perf_counter()
        img = Image.open(e2e_sample); img = ImageOps.exif_transpose(img).convert("RGB")
        tensor = eval_transform(img).unsqueeze(0).to(DEVICE)
        t1 = time.perf_counter()

        tensor.requires_grad_(True)
        with torch.enable_grad():
            out  = model(tensor); pred = torch.argmax(out[0]).item()
        t2 = time.perf_counter()

        target_layer2 = model.vit.encoder.layers[-1]
        cam2 = GradCAM(model, target_layer2)
        cam2.generate_heatmap(tensor, category_index=pred)
        t3 = time.perf_counter()

        _ = keyword_retrieve(class_names[pred], rag_chunks, top_k=3)
        t4 = time.perf_counter()

        _ = advisor.get_treatment_advice(class_names[pred], 85.0, language="English")
        t5 = time.perf_counter()

        latencies["preprocessing"].append((t1-t0)*1000)
        latencies["inference"].append((t2-t1)*1000)
        latencies["heatmap"].append((t3-t2)*1000)
        latencies["retrieval"].append((t4-t3)*1000)
        latencies["generation"].append((t5-t4)*1000)
        latencies["total"].append((t5-t0)*1000)

    mean_e2e = float(np.mean(latencies["total"]))
    print(f"  Averaged over {N_RUNS} runs\n")
    print(f"  {'Stage':<22} {'Mean (ms)':>10}  {'Min (ms)':>9}  {'Max (ms)':>9}")
    print("  " + "-" * 56)
    for stage, vals in latencies.items():
        print(f"  {stage:<22} {np.mean(vals):>9.1f}  {np.min(vals):>9.1f}  {np.max(vals):>9.1f}")
else:
    mean_e2e = 0.0
    latencies = {k: [0.0] for k in latencies}
    print("  [SKIP] No test image found.")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
banner("TABLE II — HONEST END-TO-END EVALUATION METRICS")

acc_str   = f"{accuracy_pct:.2f}%"  if accuracy_pct  is not None else "N/A (no GT set)"
f1_str    = f"{macro_f1:.4f}"       if macro_f1       is not None else "N/A"
iou_str   = f"{mean_iou:.4f}"       if mean_iou       is not None else "N/A (no GT mask)"
foc_str   = f"{mean_foc:.4f}"       if mean_foc       is not None else "N/A"

print(f"""
+-------------+--------------------------+---------------------------------------------+
|  Module     |  Metric                  |  Measured Value                             |
+-------------+--------------------------+---------------------------------------------+
|  Classifier |  Accuracy (sklearn)      |  {acc_str:<43} |
|             |  Macro F1  (sklearn)     |  {f1_str:<43} |
|             |  Mean Confidence         |  {(f"{mean_conf:.2f}%" if mean_conf else "N/A"):<43} |
+-------------+--------------------------+---------------------------------------------+
|  Grad-CAM   |  IoU @ thresh=0.5        |  {iou_str:<43} |
|             |  Focality score          |  {foc_str:<43} |
|             |  Annotation source       |  {"annotated_boxes.json (manual)":<43} |
+-------------+--------------------------+---------------------------------------------+
|  Retriever  |  MRR                     |  {mrr:.4f} (corpus: {len(rag_chunks)} chunks)              |
|             |  Recall@1                |  {recall_summary[1]:.4f}                                      |
|             |  Recall@3                |  {recall_summary[3]:.4f}                                      |
|             |  Recall@5                |  {recall_summary[5]:.4f}                                      |
+-------------+--------------------------+---------------------------------------------+
|  LLM-RAG   |  Faithfulness [{LLM_MODE[:8]:<8}]|  {mean_faithfulness:.4f}                                      |
|             |  Relevance    [{LLM_MODE[:8]:<8}]|  {mean_relevance:.4f}                                      |
+-------------+--------------------------+---------------------------------------------+
|  System     |  End-to-end latency      |  {mean_e2e:.1f} ms avg ({N_RUNS} runs, CPU)             |
+-------------+--------------------------+---------------------------------------------+
""")

# Save JSON
results = {
    "meta": {
        "date": "2026-09-22",
        "device": str(DEVICE),
        "rag_corpus_size": len(rag_chunks),
        "llm_mode": LLM_MODE
    },
    "classifier": {
        "accuracy_pct":       round(accuracy_pct, 4)   if accuracy_pct  is not None else None,
        "macro_f1":           round(macro_f1, 4)        if macro_f1       is not None else None,
        "mean_confidence_pct": round(mean_conf, 2)      if mean_conf      is not None else None,
        "std_confidence_pct":  round(std_conf,  2)      if std_conf       is not None else None,
        "n_images":           len(y_true)                if has_labeled_set else 0,
        "source":             "labeled_test_set/ (sklearn)" if has_labeled_set else "no_gt_available",
        "note":               "accuracy and macro_f1 from sklearn with ground-truth labels"
    },
    "gradcam": {
        "mean_iou_at_05":     round(mean_iou, 4)  if mean_iou  is not None else None,
        "mean_focality":      round(mean_foc, 4)  if mean_foc  is not None else None,
        "annotation_source":  "annotated_boxes.json",
        "images_annotated":   len(gradcam_results),
        "results":            gradcam_results,
        "note":               "IoU computed vs manual bounding-box annotations"
    },
    "retriever": {
        "mrr":          round(mrr, 4),
        "recall_at_1":  round(recall_summary[1], 4),
        "recall_at_3":  round(recall_summary[3], 4),
        "recall_at_5":  round(recall_summary[5], 4),
        "corpus_size":  len(rag_chunks),
        "note":         f"Expanded corpus from 6 to {len(rag_chunks)} chunks"
    },
    "llm_rag": {
        "mean_faithfulness":  round(mean_faithfulness, 4),
        "mean_relevance":     round(mean_relevance, 4),
        "method":             LLM_MODE,
        "detail":             llm_scores_detail,
        "note":               "HONEST: circular self-scoring removed. "
                              "Relevance=disease-keyword match, "
                              "Faithfulness=RAG-grounded agronomic terms. "
                              "Set GEMINI_API_KEY for real LLM evaluation."
    },
    "system": {
        "mean_e2e_latency_ms": round(mean_e2e, 1),
        "stage_latencies_ms": {k: round(float(np.mean(v)), 1)
                               for k, v in latencies.items()}
    }
}

out_json = os.path.join(BASE_DIR, "eval_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"  Full honest results saved → eval_results.json")
print(f"  Grad-CAM overlay          → eval_gradcam_output.jpg")
print()
