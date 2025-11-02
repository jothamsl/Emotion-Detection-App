# Deployment Guide for Render

This guide will help you deploy your Emotion Detection Flask app to Render, a modern cloud platform that makes deployment simple and fast.

## 🚀 Quick Deploy to Render

### Prerequisites
- A GitHub account
- Your code pushed to a GitHub repository
- A Render account (free at [render.com](https://render.com))

### Step 1: Prepare Your Repository

1. **Push your code to GitHub**:
   ```bash
   # Initialize git repository (if not already done)
   git init
   
   # Add all files
   git add .
   
   # Commit your changes
   git commit -m "Initial commit - Emotion Detection App"
   
   # Add your GitHub repository as origin
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   
   # Push to GitHub
   git push -u origin main
   ```

2. **Verify deployment files are included**:
   - ✅ `Procfile` - Tells Render how to start your app
   - ✅ `requirements.txt` - Lists Python dependencies
   - ✅ `render.yaml` - Render configuration (optional)
   - ✅ `.gitignore` - Prevents uploading unnecessary files

### Step 2: Deploy on Render

#### Method 1: One-Click Deploy (Recommended)

1. **Go to Render Dashboard**:
   - Visit [render.com](https://render.com)
   - Sign up/Log in with GitHub

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select your emotion detection app repository

3. **Configure Service**:
   ```
   Name: emotion-detection-app
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app:app
   ```

4. **Advanced Settings** (Optional):
   ```
   Instance Type: Free
   Environment Variables:
     - PYTHON_VERSION: 3.11.0
     - FLASK_ENV: production
   ```

5. **Click "Create Web Service"**

#### Method 2: Using render.yaml (Infrastructure as Code)

If you have the `render.yaml` file in your repository:

1. Go to Render Dashboard
2. Click "New +" → "Blueprint"
3. Connect your repository
4. Render will automatically use the `render.yaml` configuration

### Step 3: Monitor Deployment

1. **Watch Build Logs**:
   - Render will show real-time build logs
   - First deployment takes 5-10 minutes (model download)
   - Look for: "Your service is live 🎉"

2. **Common Build Process**:
   ```
   ==> Installing dependencies from requirements.txt
   ==> Downloading AI model (this takes a few minutes)
   ==> Starting gunicorn server
   ==> Deploy succeeded! 🎉
   ```

3. **Access Your App**:
   - Render provides a free URL: `https://your-app-name.onrender.com`
   - Click the link to test your deployed app

## 🔧 Configuration Details

### Environment Variables

Render automatically sets these for you:

- `PORT` - The port your app should listen on
- `PYTHON_VERSION` - Python version to use
- `FLASK_ENV` - Set to "production" for deployment

### Custom Environment Variables (Optional)

You can add these in Render Dashboard → Environment:

```
FLASK_SECRET_KEY=your-secret-key-here
MAX_UPLOAD_SIZE=16777216
```

### Database

Your app uses SQLite, which works perfectly on Render:
- Database file is created automatically
- Persists between deployments
- No additional setup needed

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. **Build Fails - "No module named 'torch'"**
```
Solution: Make sure requirements.txt includes all dependencies
Check: pip install -r requirements.txt works locally
```

#### 2. **App Crashes - "Model Download Failed"**
```
Solution: First deployment takes longer due to model download
Wait: 10-15 minutes for complete deployment
Check: Build logs for download progress
```

#### 3. **Port Already in Use Error**
```
Solution: Remove hardcoded port from app.py
Ensure: app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
```

#### 4. **File Upload Errors**
```
Solution: Render has disk space limits
Check: Upload files are temporary and cleaned up
Verify: MAX_CONTENT_LENGTH is set appropriately
```

#### 5. **Slow Response Times**
```
Free Tier: Apps sleep after 15 minutes of inactivity
First Request: May take 30-60 seconds to wake up
Solution: Upgrade to paid tier for always-on service
```

### Debugging Commands

Check your deployment:

```bash
# Test your deployed app
curl https://your-app-name.onrender.com/health

# Expected response:
{
  "status": "healthy",
  "model_loaded": true,
  "database_predictions": 0,
  "timestamp": "2024-01-15T10:30:00"
}
```

## 📊 Performance Tips

### Free Tier Limitations
- **Sleep Mode**: App sleeps after 15 minutes of inactivity
- **Memory**: 512 MB RAM
- **Storage**: 1 GB disk space
- **Bandwidth**: 100 GB/month

### Optimization for Free Tier
1. **Model Caching**: Model downloads once and is cached
2. **Database**: SQLite is lightweight and fast
3. **Static Files**: Served efficiently by Flask
4. **Compression**: Gzip enabled automatically

### Upgrading (Optional)
For production use, consider:
- **Starter Plan ($7/month)**: Always-on, more resources
- **Standard Plan ($25/month)**: Higher performance, better reliability

## 🔗 Useful Links

### Your Deployed App URLs
- **Main App**: `https://your-app-name.onrender.com`
- **Health Check**: `https://your-app-name.onrender.com/health`
- **API Stats**: `https://your-app-name.onrender.com/stats`

### Render Dashboard
- **Logs**: Monitor real-time application logs
- **Metrics**: View performance and usage statistics
- **Settings**: Update environment variables and configuration

## 🎯 Post-Deployment Checklist

After successful deployment:

- [ ] Test image upload functionality
- [ ] Verify emotion detection works
- [ ] Check database is logging predictions
- [ ] Test API endpoints (/health, /stats)
- [ ] Monitor build logs for any warnings
- [ ] Share your app URL with others!

## 🚀 Updating Your App

To deploy updates:

1. **Make changes locally**
2. **Test locally**: `python run.py`
3. **Commit and push**:
   ```bash
   git add .
   git commit -m "Update: your changes here"
   git push origin main
   ```
4. **Render auto-deploys** from your main branch

## 💡 Pro Tips

1. **Custom Domain**: Connect your own domain in Render settings
2. **SSL Certificate**: Automatically provided by Render
3. **CDN**: Static files served via global CDN
4. **Monitoring**: Set up health checks and alerts
5. **Backup**: Database automatically backed up on paid plans

## 🎉 Success!

Once deployed, your Emotion Detection app will be live at:
`https://your-app-name.onrender.com`

Share it with friends and family to detect emotions from their photos! 🎭✨

---

### Need Help?

- **Render Docs**: [docs.render.com](https://docs.render.com)
- **Community**: [community.render.com](https://community.render.com)
- **Support**: Available through Render dashboard

**Happy Deploying!** 🚀