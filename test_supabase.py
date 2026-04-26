from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Test 1: Can we connect and read from swings table
try:
    result = supabase.table("swings").select("id").limit(1).execute()
    print("✓ Supabase connection working")
    print("✓ Swings table exists")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    exit(1)

# Test 2: Can we insert a test swing
try:
    result = supabase.table("swings").insert({
        "user_ip": "test",
        "score": 99,
        "grade": "A",
        "handedness": "right",
        "in_envelope_pct": 95.0,
        "deviations": [],
        "claude_feedback": {"test": True}
    }).execute()
    print("✓ Test swing inserted successfully")

    # Clean up test row
    supabase.table("swings").delete().eq("user_ip", "test").execute()
    print("✓ Test row cleaned up")
except Exception as e:
    print(f"✗ Insert failed: {e}")
    exit(1)

# Test 3: Can we read coaching_cache table
try:
    result = supabase.table("coaching_cache").select("id").limit(1).execute()
    print("✓ Coaching cache table exists")
except Exception as e:
    print(f"✗ Coaching cache table missing: {e}")
    exit(1)

# Test 4: Can we read daily_usage table
try:
    result = supabase.table("daily_usage").select("*").limit(1).execute()
    print("✓ Daily usage table exists")
except Exception as e:
    print(f"✗ Daily usage table missing: {e}")
    exit(1)

print("\n All Supabase tests passed. Ready to proceed.")
