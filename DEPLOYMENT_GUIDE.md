# 🚀 El GPT — 100% Free Cloud Deployment Guide

This guide walks you through deploying **El GPT** online so anyone in the world can chat with it on the web 24/7 for **100% free** (no credit card required).

---

## ⚡ Step 1: Get Your Free Groq API Key (1 minute)

1. Open **[console.groq.com](https://console.groq.com)** in your browser.
2. Sign in with Google or GitHub (free, no payment method needed).
3. In the left navigation, click **API Keys**.
4. Click **Create API Key**, name it `El-GPT`, and click **Create**.
5. Copy your key (starts with `gsk_...`).

> **Tip**: You can test this key locally right now! Open your local El GPT interface (`http://localhost:8000`), click the **Cloud ⚡** button in the top navbar, paste your key, and click **Save API Key**.

---

## 📦 Step 2: Push Your Project to GitHub (Free)

1. Open your Mac Terminal and navigate to this folder:
   ```bash
   cd "/Users/lucas/El GPT"
   ```

2. Initialize Git (your `.gitignore` already protects your heavy model files and keys):
   ```bash
   git init
   git add .
   git commit -m "Add cloud engine and deployment configuration"
   ```

3. Go to **[github.com](https://github.com)** and create a new repository named `el-gpt`.
4. Link and push your code:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/el-gpt.git
   git push -u origin main
   ```

---

## 🌐 Step 3: Deploy to Render.com (100% Free 24/7 Web Hosting)

1. Go to **[render.com](https://render.com)** and sign up with GitHub.
2. In the Render Dashboard, click **New +** &rarr; **Web Service**.
3. Choose **Build and deploy from a Git repository** and connect your `el-gpt` repo.
4. Set the following configuration:
   * **Name**: `el-gpt` (or your chosen name)
   * **Language**: `Python 3`
   * **Branch**: `main`
   * **Build Command**:
     ```bash
     pip install -r requirements-cloud.txt
     ```
   * **Start Command**:
     ```bash
     uvicorn server.app:app --host 0.0.0.0 --port $PORT
     ```
   * **Instance Type**: Select **Free** ($0/month).

5. Scroll down to **Environment Variables**, click **Add Environment Variable**:
   * **Key**: `GROQ_API_KEY`
   * **Value**: *(Paste your `gsk_...` key from Step 1)*

6. Click **Deploy Web Service** at the bottom.

In ~1–2 minutes, Render will build your site and generate your live public link:
👉 **`https://el-gpt.onrender.com`**

Share this link with anyone! They can open it on their iPhone, Android, or laptop and chat with zero lag.

---

## ⚡ Instant Shortcut: Share from your Mac Right Now (Cloudflare Tunnel)

If you want to immediately let a friend test from their phone without setting up GitHub:
1. Make sure your server is running (`./run.sh` or `uvicorn server.app:app`).
2. In a new Terminal window, run:
   ```bash
   npx cloudflared tunnel --url http://localhost:8000
   ```
3. Copy the temporary HTTPS URL shown in the terminal (e.g. `https://example-words.trycloudflare.com`).
4. Anyone who opens that link will see your ChatGPT UI live!
