# CampusFit Deployment (Flask + Docker + Render)

## 1. GitHub
1. Create a GitHub repository (e.g. `CampusFit`).
2. Upload all files and folders in this project directory, including the CSV, notebook, `app.py`, `templates`, `Dockerfile`, `requirements.txt`, and `render.yaml`.

## 2. Deploy on Render
1. Sign in at https://render.com and choose **New > Web Service**.
2. Connect your GitHub repository.
3. Select **Docker** as the environment (Render detects the Dockerfile).
4. Click **Deploy**. Wait for the build to finish, then open the Render URL.

Alternatively, the included `render.yaml` can be used with Render Blueprint setup.

## 3. Run locally (optional)
Install Docker Desktop, open a terminal in this folder and run:

```bash
docker build -t campusfit .
docker run -p 10000:10000 campusfit
```

Then visit http://localhost:10000.

The deployment trains its own lightweight Decision Tree from the included CSV at startup; the original notebook and its saved files are kept. This web app is a deployment companion, not a replacement for the notebook experiments.
