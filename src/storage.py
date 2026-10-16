"""
Cloud storage layer for contributed training images.

Priority order:
  1. Supabase Storage  (if SUPABASE_URL + SUPABASE_KEY are set)
  2. Local disk        (public/contributions/ — fallback for dev)
"""

import os
import uuid
from pathlib import Path

SUPABASE_URL    = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY    = os.environ.get("SUPABASE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "training-contributions")
BASE_URL        = os.environ.get("BASE_URL", "http://localhost:8000")

# Local fallback directory
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_CONTRIB_DIR = os.path.join(_BASE_DIR, "public", "contributions")
os.makedirs(LOCAL_CONTRIB_DIR, exist_ok=True)


def _supabase_client():
    """Return a Supabase client if credentials are configured, else None."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except ImportError:
        print("[storage] supabase-py not installed — using local disk fallback.")
        return None
    except Exception as e:
        print(f"[storage] Supabase client error: {e} — using local disk fallback.")
        return None


def upload_image(file_bytes: bytes, original_filename: str) -> tuple[str, str]:
    """
    Upload a contributed training image.

    Returns
    -------
    (public_url, storage_path)
      public_url   — accessible HTTP URL for the image
      storage_path — internal key / absolute path (used for download)
    """
    ext      = Path(original_filename).suffix.lower() or ".jpg"
    filename = f"{uuid.uuid4()}{ext}"
    key      = f"contributions/{filename}"

    client = _supabase_client()
    if client:
        try:
            client.storage.from_(SUPABASE_BUCKET).upload(key, file_bytes)
            public_url = client.storage.from_(SUPABASE_BUCKET).get_public_url(key)
            print(f"[storage] Uploaded to Supabase: {key}")
            return public_url, key
        except Exception as e:
            print(f"[storage] Supabase upload failed ({e}). Falling back to local disk.")

    # Local fallback
    local_path = os.path.join(LOCAL_CONTRIB_DIR, filename)
    with open(local_path, "wb") as f:
        f.write(file_bytes)
    public_url = f"{BASE_URL}/contributions/{filename}"
    print(f"[storage] Saved locally: {local_path}")
    return public_url, local_path


def download_image(storage_path: str) -> bytes | None:
    """
    Download a contributed image by its storage_path.
    Tries Supabase first, then local disk.
    """
    # Absolute local path
    if os.path.isabs(storage_path) and os.path.exists(storage_path):
        with open(storage_path, "rb") as f:
            return f.read()

    # Supabase key (relative path like "contributions/uuid.jpg")
    client = _supabase_client()
    if client:
        try:
            data = client.storage.from_(SUPABASE_BUCKET).download(storage_path)
            return data
        except Exception as e:
            print(f"[storage] Supabase download failed ({e}).")

    # Last resort: check local dir by filename
    filename = os.path.basename(storage_path)
    local_path = os.path.join(LOCAL_CONTRIB_DIR, filename)
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            return f.read()

    print(f"[storage] Could not find image: {storage_path}")
    return None
