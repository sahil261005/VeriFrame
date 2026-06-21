import os
import sys
import time

# add current directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from main import app
import database

# make sure we use a clean test database file
# so we don't dirty the dev database
os.environ["DATABASE_URL"] = "sqlite:///./test_veriframe.db"

# recreate tables
database.create_tables()

client = TestClient(app)


def test_auth_and_upload():
    print("--- 1. Testing Registration ---")
    reg_response = client.post(
        "/auth/register",
        json={"email": "testuser@example.com", "password": "securepassword123"}
    )
    print("Registration response:", reg_response.status_code, reg_response.json())
    assert reg_response.status_code == 200
    token = reg_response.json()["access_token"]
    
    print("\n--- 2. Testing Login ---")
    login_response = client.post(
        "/auth/login",
        json={"email": "testuser@example.com", "password": "securepassword123"}
    )
    print("Login response:", login_response.status_code, login_response.json())
    assert login_response.status_code == 200
    
    headers = {"Authorization": f"Bearer {token}"}
    
    print("\n--- 3. Testing Uploading Video ---")
    # we'll upload sample.mp4 from the root workspace
    video_path = "../sample.mp4"
    if not os.path.exists(video_path):
        print(f"error: {video_path} not found. skipping upload test.")
        return
        
    with open(video_path, "rb") as f:
        upload_response = client.post(
            "/upload",
            headers=headers,
            files={"file": ("sample.mp4", f, "video/mp4")}
        )
    print("Upload response:", upload_response.status_code, upload_response.json())
    assert upload_response.status_code == 200
    job_id = upload_response.json()["id"]
    
    print("\n--- 4. Polling Status ---")
    # since TestClient is synchronous and runs BackgroundTasks in the same thread
    # at the end of the request, the background task has already started/finished!
    # Let's poll or check status
    for i in range(5):
        status_response = client.get(f"/analysis/{job_id}", headers=headers)
        status_data = status_response.json()
        print(f"Poll {i+1}: status = {status_data['status']}, verdict = {status_data.get('final_verdict')}")
        if status_data["status"] in ["completed", "failed"]:
            break
        time.sleep(1)
        
    print("\n--- 5. Testing History ---")
    history_response = client.get("/history", headers=headers)
    print("History count:", len(history_response.json()))
    assert history_response.status_code == 200
    
    print("\n--- 6. Testing PDF Report Download ---")
    report_response = client.get(f"/report/{job_id}/pdf", headers=headers)
    print("PDF Response status:", report_response.status_code)
    print("Content-Type:", report_response.headers.get("content-type"))
    assert report_response.status_code == 200
    
    print("\n--- Cleanup ---")
    # clean up the test database file
    db_file = "./test_veriframe.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        print("deleted test database.")
        
    print("\nAPI test completed successfully!")


if __name__ == "__main__":
    test_auth_and_upload()
