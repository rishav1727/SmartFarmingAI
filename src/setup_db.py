"""
SmartFarmingAI — Cloud Database Setup & Verification Tool

Usage:
    python -m src.setup_db

This utility:
  1. Inspects DATABASE_URL (supports PostgreSQL from Supabase / Render, or local SQLite).
  2. Tests engine connection and latency.
  3. Creates all required ORM tables (contributions, model_versions).
  4. Seeds initial baseline ModelVersion record if empty.
  5. Validates Cloud Storage (Supabase bucket or local disk fallback).
"""

import os
import sys
import time
from datetime import datetime

# Ensure project root is in sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

from src.database import engine, SessionLocal, init_db, Contribution, ModelVersion, DATABASE_URL
from src.storage import _supabase_client, SUPABASE_BUCKET, SUPABASE_URL


def run_setup():
    print("=" * 60)
    print("  SmartFarmingAI — Database & Storage Initializer")
    print("=" * 60)

    # 1. Inspect DB Connection
    db_type = "PostgreSQL (Cloud)" if "postgresql" in DATABASE_URL or "postgres" in DATABASE_URL else "SQLite (Local)"
    safe_url = DATABASE_URL
    if "@" in safe_url:
        # Mask password in URL
        proto_user = safe_url.split("@")[0].split("://")
        proto = proto_user[0]
        host = safe_url.split("@")[1]
        safe_url = f"{proto}://*****:*****@{host}"

    print(f"\n[1/4] Checking Database Target...")
    print(f"      Type : {db_type}")
    print(f"      URL  : {safe_url}")

    # 2. Test Connection & Create Tables
    print(f"\n[2/4] Connecting to Database & Initializing Tables...")
    start_t = time.time()
    try:
        with engine.connect() as conn:
            pass
        latency = (time.time() - start_t) * 1000
        print(f"      Connection Successful! Latency: {latency:.1f}ms")
    except Exception as e:
        print(f"      ERROR connecting to database: {e}")
        return False

    try:
        init_db()
        print("      Schema verified: 'contributions' and 'model_versions' tables ready.")
    except Exception as e:
        print(f"      ERROR creating tables: {e}")
        return False

    # 3. Seed Baseline Model Version
    print(f"\n[3/4] Verifying Seed Data...")
    session = SessionLocal()
    try:
        existing_version = session.query(ModelVersion).filter(ModelVersion.version == "v1.0-base").first()
        if not existing_version:
            baseline = ModelVersion(
                version="v1.0-base",
                accuracy=98.2,
                val_loss=0.065,
                trained_on_samples=54305,
                new_contributions=0,
                weights_filename="best_model.pth",
                is_active=True,
                notes="Initial Vision Transformer (ViT-B/16) checkpoint trained on PlantVillage 38 classes.",
                created_at=datetime.utcnow()
            )
            session.add(baseline)
            session.commit()
            print("      Baseline model record seeded: 'v1.0-base' (Active).")
        else:
            print(f"      Baseline model record already exists: '{existing_version.version}' (Active={existing_version.is_active}).")

        total_contribs = session.query(Contribution).count()
        total_versions = session.query(ModelVersion).count()
        print(f"      Current Records -> Contributions: {total_contribs}, Model Versions: {total_versions}")
    except Exception as e:
        print(f"      Warning while seeding: {e}")
        session.rollback()
    finally:
        session.close()

    # 4. Storage Validation
    print(f"\n[4/4] Validating Storage Layer...")
    supabase_cli = _supabase_client()
    if supabase_cli:
        print(f"      Supabase Client: Connected ({SUPABASE_URL})")
        print(f"      Storage Bucket : {SUPABASE_BUCKET}")
        try:
            buckets = supabase_cli.storage.list_buckets()
            bucket_names = [b.name if hasattr(b, 'name') else b.get('name') for b in buckets]
            if SUPABASE_BUCKET in bucket_names:
                print(f"      Bucket Status  : Verified '{SUPABASE_BUCKET}' exists in Supabase.")
            else:
                print(f"      Bucket Notice  : '{SUPABASE_BUCKET}' not found in bucket list.")
                print(f"                       Available buckets: {bucket_names}")
                print(f"                       Please create a public bucket named '{SUPABASE_BUCKET}' in Supabase Dashboard.")
        except Exception as e:
            print(f"      Bucket Check   : Notice: {e}")
    else:
        print("      Supabase: Not configured (SUPABASE_URL/KEY not set).")
        print("      Using Local Storage: 'public/contributions/' (Ideal for local development).")

    print("\n" + "=" * 60)
    print("  Database & Storage Setup Completed Successfully!")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    success = run_setup()
    sys.exit(0 if success else 1)
