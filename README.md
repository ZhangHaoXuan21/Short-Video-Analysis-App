# 🎬 Short Video Analyst (Local GenAI Desktop App)

An **offline, agentic AI video analysis system** built with **Streamlit + LangGraph + LangChain + Hugging Face models**.  
This project was developed as part of the **GenAI Software Solutions Engineer Test Assignment**.

---

## 🚀 Overview

**Short Video Analyst** is a fully local AI desktop application that allows users to:

- 🧠 Analyze and query short `.mp4` videos through natural language
- 🗣️ Transcribe audio and detect key discussion topics
- 👁️ Recognize objects, scenes, and text in video frames
- 🧾 Generate PDF and PowerPoint summaries locally
- 💬 Maintain persistent chat memory even after restart

Everything runs **locally**, with **no cloud inference or external APIs**.

---

## 🧩 AI Agents Architecture
![](demo_images/langgraph_ai_agents_architecture.png)


## 🧠 Core Technologies

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Backend Framework** | LangGraph + LangChain |
| **LLM / VLM Models** | Qwen1.7B, SmolVLM2, Whisper Tiny |
| **Report Generation** | ReportLab (PDF) + python-pptx |
| **Persistence** | Pickle-based chat memory |
| **Environment** | Conda-managed virtual environment |

---

## 🧪 Features

✅ Upload short video clips (≤ 1 minute)  
✅ Natural language queries like:
   - “Transcribe the video”
   - “What objects are shown?”
   - “Create a PowerPoint of the key points”  
✅ Multi-agent orchestration for transcription, vision, and generation  
✅ Offline inference with local Hugging Face models  
✅ Persistent memory store for previous chat sessions  
✅ Output generation: PDF & PPTX


## 🧰 Setup Guide

### 1️⃣ Clone Repository
- git clone https://github.com/ZhangHaoXuan21/Short-Video-Analysis-App.git
- cd Short-Video-Analysis-App

### 2️⃣ Setup Conda Environment
conda env create -f environment.yml -n <new_environment_name>

### 3️⃣ Run the App
streamlit run app.py


## 🧱 Example Workflow

### Example 1️⃣
1. Upload a short video
![](demo_images/demo_1/upload.png)

2. Get the video transcription
![](demo_images/demo_1/transcript.png)

3. Perform Video Analysis
![](demo_images/demo_1/video.png)

4. Generate PPTX File
![](demo_images/demo_1/pptx.png)

5. Generate PDF File
- Sometimes, the generation will fail due to wrong format (e.g: generate pptx instead of pdf), multiple retries are needed.
![](demo_images/demo_1/pdf.png)


💡 Limitations & Future Work

- Model inference speed could be improved with OpenVINO optimization.
- Current Streamlit frontend could be ported to Tauri + React for production.
- Multi-agent coordination could later use gRPC or a proper MCP server for modularity.
- Future versions may integrate audio emotion analysis and topic clustering.

