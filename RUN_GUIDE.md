# How to Run FinSense Project

## Quick Start (5 Minutes)

### Step 1: Check Python Version

Open terminal/command prompt and type:
```bash
python --version
```

**You need Python 3.10 or higher!**

If you don't have Python:
- Download from [python.org](https://www.python.org/downloads/)
- **Important:** Check "Add Python to PATH" during installation

---

### Step 2: Navigate to Project Folder

```bash
cd C:\Users\falgu\OneDrive\Desktop\FinSense
```

---

### Step 3: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**What this does:** Creates an isolated environment so your project doesn't conflict with other Python projects.

**You'll see `(venv)` in your terminal** - that means it's working!

---

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**What this does:** Downloads and installs all the libraries your project needs.

**This takes 5-10 minutes** - grab a coffee! ☕

---

### Step 5: Set Up Environment Variables

**Create `.env` file:**

1. Copy the example file:
   ```bash
   copy config\env.example .env
   ```
   (Mac/Linux: `cp config/env.example .env`)

2. Open `.env` file in Notepad/TextEdit

3. Add your OpenAI API key (optional, only needed for chatbot):
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

**Note:** You can run the project without API key, but chatbot features won't work.

---

## Running the Project

### Option 1: Run Backend API (Recommended First)

**Open a terminal and run:**
```bash
python run_backend.py
```

**You'll see:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**✅ Backend is running!** Open browser: `http://localhost:8000`

**Test it:** Go to `http://localhost:8000/docs` - you'll see the API documentation!

---

### Option 2: Run Streamlit Dashboard

**Open a NEW terminal** (keep backend running in first terminal):

```bash
# Activate virtual environment again
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Mac/Linux

# Run dashboard
streamlit run dashboard/app.py
```

**You'll see:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

**✅ Dashboard is running!** It will open automatically in your browser.

---

### Option 3: Run Both Together

**Windows:**
```bash
start.bat backend    # Terminal 1
start.bat dashboard  # Terminal 2
```

**Mac/Linux:**
```bash
./start.sh both
```

---

## What Each Component Does

### 1. Backend API (`run_backend.py`)
- **Port:** 8000
- **URL:** http://localhost:8000
- **What it does:** Provides API endpoints for predictions, risk analysis, and explanations
- **API Docs:** http://localhost:8000/docs

### 2. Streamlit Dashboard (`dashboard/app.py`)
- **Port:** 8501
- **URL:** http://localhost:8501
- **What it does:** Interactive web interface to visualize data, make predictions, analyze risk, and chat with AI

---

## Testing the Project

### Test Backend API

1. **Health Check:**
   - Open: http://localhost:8000/health
   - Should show: `{"status": "healthy"}`

2. **Get Live Price:**
   - Open: http://localhost:8000/predict/live/AAPL
   - Should show: Current price data

3. **Risk Analysis:**
   - Open: http://localhost:8000/risk/AAPL?period=1y
   - Should show: Risk metrics

### Test Dashboard

1. Open: http://localhost:8501
2. Enter a stock symbol (e.g., "AAPL")
3. Click "Load Data"
4. Explore different pages:
   - **Dashboard:** View stock charts
   - **Forecasting:** Make predictions
   - **Risk Analysis:** See risk metrics
   - **Live & Chatbot:** Real-time data and AI chat

---

## Common Issues & Solutions

### Issue 1: "Python is not recognized"

**Solution:**
- Reinstall Python and check "Add Python to PATH"
- Or use full path: `C:\Python310\python.exe --version`

### Issue 2: "pip is not recognized"

**Solution:**
```bash
python -m pip install -r requirements.txt
```

### Issue 3: "Port 8000 already in use"

**Solution:**
- Close other applications using port 8000
- Or change port in `run_backend.py`:
  ```python
  uvicorn.run(..., port=8001)  # Use different port
  ```

### Issue 4: "Module not found"

**Solution:**
- Make sure virtual environment is activated (you see `(venv)`)
- Reinstall dependencies: `pip install -r requirements.txt`

### Issue 5: "OpenAI API key error"

**Solution:**
- This is OK! Project works without it
- Only chatbot features need API key
- To use chatbot: Get key from [platform.openai.com](https://platform.openai.com/api-keys)

### Issue 6: "TensorFlow/GPU errors"

**Solution:**
- Install CPU-only version:
  ```bash
  pip install tensorflow-cpu
  ```

---

## Running with Docker (Alternative)

If you have Docker installed:

```bash
# Build image
docker build -t finsense .

# Run backend
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key finsense python run_backend.py

# Run dashboard
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key finsense streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0
```

Or use Docker Compose:
```bash
docker-compose up
```

---

## Project Structure Overview

```
FinSense/
├── run_backend.py          # Start backend API
├── run_dashboard.py        # Start dashboard
├── dashboard/app.py         # Main dashboard code
├── backend/main.py          # Backend API code
├── requirements.txt         # All dependencies
├── config/env.example      # Environment variables template
└── .env                    # Your API keys (create this)
```

---

## Step-by-Step Checklist

- [ ] Python 3.10+ installed
- [ ] Navigated to project folder
- [ ] Created virtual environment (`python -m venv venv`)
- [ ] Activated virtual environment (see `(venv)` in terminal)
- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Created `.env` file (optional, for chatbot)
- [ ] Started backend (`python run_backend.py`)
- [ ] Started dashboard (`streamlit run dashboard/app.py`)
- [ ] Opened browser to http://localhost:8501

---

## Quick Commands Reference

| Task | Command |
|------|---------|
| Activate venv (Windows) | `venv\Scripts\activate` |
| Activate venv (Mac/Linux) | `source venv/bin/activate` |
| Install dependencies | `pip install -r requirements.txt` |
| Run backend | `python run_backend.py` |
| Run dashboard | `streamlit run dashboard/app.py` |
| Check Python version | `python --version` |
| Check installed packages | `pip list` |

---

## Need Help?

1. **Check logs:** Look in `logs/` folder for error messages
2. **Read documentation:** See `README.md` and `QUICKSTART.md`
3. **Common errors:** See "Common Issues" section above

---

## Example: Complete First Run

```bash
# 1. Go to project folder
cd C:\Users\falgu\OneDrive\Desktop\FinSense

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
venv\Scripts\activate

# 4. Install everything
pip install -r requirements.txt

# 5. Create .env file (optional)
copy config\env.example .env

# 6. Run backend (Terminal 1)
python run_backend.py

# 7. Open NEW terminal, activate venv, run dashboard (Terminal 2)
venv\Scripts\activate
streamlit run dashboard/app.py
```

**That's it! Your project is running! 🎉**

---

## What to Do Next?

1. **Explore Dashboard:** Try different stock symbols (AAPL, MSFT, GOOGL)
2. **Test Predictions:** Use the Forecasting page
3. **Analyze Risk:** Check Risk Analysis page
4. **Try Chatbot:** Ask questions about financial concepts (if API key set)

**Enjoy your Financial Intelligence System! 📈**













