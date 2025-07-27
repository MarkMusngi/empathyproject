#!/usr/bin/env python3
"""
Standalone Ollama Connection Test Script
Place this file in your Django project root directory and run: python test_ollama.py
"""

import requests
import time

def test_ollama_connection():
    """Test Ollama connection and performance"""
    
    print("🔍 Testing Ollama Connection...")
    
    # Test 1: Basic connectivity
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        print(f"✅ Ollama is running (Status: {response.status_code})")
        
        models = response.json().get('models', [])
        print(f"📋 Available models: {[m['name'] for m in models]}")
        
        if not any('mistral' in m['name'] for m in models):
            print("❌ Mistral model not found!")
            return False
            
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False
    
    # Test 2: Simple query speed
    print("\n⏱️ Testing response speed...")
    
    test_payload = {
        "model": "mistral",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False
    }
    
    start_time = time.time()
    try:
        response = requests.post("http://localhost:11434/api/chat", 
                               json=test_payload, timeout=15)
        end_time = time.time()
        
        if response.status_code == 200:
            print(f"✅ Simple query completed in {end_time - start_time:.2f} seconds")
        else:
            print(f"❌ Query failed with status: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Simple query timed out - Ollama is struggling!")
        return False
    except Exception as e:
        print(f"❌ Query error: {e}")
        return False
    
    # Test 3: Interview-like query
    print("\n📝 Testing interview query...")
    
    interview_payload = {
        "model": "mistral", 
        "messages": [{"role": "user", "content": "Give brief feedback on this interview answer: I worked on a team project and it went well."}],
        "stream": False
    }
    
    start_time = time.time()
    try:
        response = requests.post("http://localhost:11434/api/chat",
                               json=interview_payload, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            print(f"✅ Interview query completed in {end_time - start_time:.2f} seconds")
            result = response.json()
            print(f"📄 Response length: {len(result['message']['content'])} characters")
            return True
        else:
            print(f"❌ Interview query failed: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Interview query timed out!")
        return False
    except Exception as e:
        print(f"❌ Interview query error: {e}")
        return False

if __name__ == "__main__":
    success = test_ollama_connection()
    
    if not success:
        print("\n🔧 Troubleshooting suggestions:")
        print("1. Restart Ollama: Close terminal, then run 'ollama serve'")
        print("2. Check model: Run 'ollama pull mistral'") 
        print("3. Check resources: Close other heavy applications")
        print("4. Try smaller model: 'ollama pull llama3.2:1b' (faster)")
    else:
        print("\n✅ Ollama is working properly!")