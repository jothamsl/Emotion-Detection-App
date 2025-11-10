#!/bin/bash

# GitHub Deployment Script for Emotion Detection Web App
# =====================================================
# Author: SOBOYEJO-OLUWALASE_23CD034363
# Purpose: Deploy emotion detection app to GitHub for Render hosting

set -e  # Exit on any error

echo "🚀 GitHub Deployment Script for Emotion Detection Web App"
echo "=========================================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project information
PROJECT_NAME="emotion-detection-web-app-soboyejo"
STUDENT_ID="SOBOYEJO-OLUWALASE_23CD034363"
DESCRIPTION="AI-powered emotion detection web application with live camera and image upload capabilities"

echo -e "${BLUE}📦 Project: $PROJECT_NAME${NC}"
echo -e "${BLUE}👤 Student: $STUDENT_ID${NC}"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git is not installed. Please install Git first.${NC}"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ] || [ ! -f "model.py" ]; then
    echo -e "${RED}❌ Error: Not in the correct project directory!${NC}"
    echo "Please run this script from the SOBOYEJO-OLUWALASE_23CD034363_EMOTION_DETECTION_WEB_APP directory"
    exit 1
fi

echo -e "${GREEN}✅ Git is installed${NC}"
echo -e "${GREEN}✅ In correct project directory${NC}"

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo ""
    echo -e "${YELLOW}🔧 Initializing Git repository...${NC}"
    git init
    echo -e "${GREEN}✅ Git repository initialized${NC}"
else
    echo -e "${GREEN}✅ Git repository already exists${NC}"
fi

# Check if remote origin exists
if git remote get-url origin &> /dev/null; then
    echo -e "${YELLOW}⚠️  Remote 'origin' already exists. Removing and re-adding...${NC}"
    git remote remove origin
fi

# Prompt for GitHub username
echo ""
echo -e "${YELLOW}📝 Please provide your GitHub information:${NC}"
read -p "Enter your GitHub username: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo -e "${RED}❌ GitHub username is required!${NC}"
    exit 1
fi

# Construct repository URL
REPO_URL="https://github.com/$GITHUB_USERNAME/$PROJECT_NAME.git"

echo ""
echo -e "${BLUE}🔗 Repository URL: $REPO_URL${NC}"

# Add remote origin
echo ""
echo -e "${YELLOW}🔧 Adding GitHub remote...${NC}"
git remote add origin $REPO_URL

# Configure git user if not already configured
if [ -z "$(git config --global user.name)" ]; then
    read -p "Enter your Git username: " GIT_USERNAME
    git config --global user.name "$GIT_USERNAME"
fi

if [ -z "$(git config --global user.email)" ]; then
    read -p "Enter your Git email: " GIT_EMAIL
    git config --global user.email "$GIT_EMAIL"
fi

# Add all files to staging
echo ""
echo -e "${YELLOW}📦 Adding files to Git...${NC}"
git add .

# Show what will be committed
echo ""
echo -e "${BLUE}📋 Files to be committed:${NC}"
git status --porcelain

# Commit changes
echo ""
echo -e "${YELLOW}💾 Committing changes...${NC}"
COMMIT_MESSAGE="Initial deployment of emotion detection web app

- Flask web application with emotion detection
- Live camera capture and image upload functionality
- SQLite database for storing predictions
- Bootstrap responsive UI
- Hugging Face transformer model integration
- Production-ready configuration for Render deployment

Student: $STUDENT_ID
Date: $(date '+%Y-%m-%d %H:%M:%S')"

git commit -m "$COMMIT_MESSAGE"

echo -e "${GREEN}✅ Changes committed successfully${NC}"

# Push to GitHub
echo ""
echo -e "${YELLOW}🚀 Pushing to GitHub...${NC}"
echo -e "${BLUE}Note: You may be prompted for your GitHub credentials${NC}"

# Check if main branch exists, if not create it
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    git branch -M main
fi

# Push to GitHub
if git push -u origin main; then
    echo ""
    echo -e "${GREEN}🎉 SUCCESS! Code pushed to GitHub${NC}"
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ GITHUB DEPLOYMENT COMPLETE${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}📝 Next Steps for Render Deployment:${NC}"
    echo ""
    echo "1. 🌐 Go to https://render.com"
    echo "2. 🔐 Sign up/Login (connect with GitHub for easier deployment)"
    echo "3. ➕ Click 'New +' → 'Web Service'"
    echo "4. 🔗 Connect your GitHub repository: $GITHUB_USERNAME/$PROJECT_NAME"
    echo "5. ⚙️  Configure deployment settings:"
    echo "   • Name: emotion-detection-web-app-soboyejo"
    echo "   • Environment: Python"
    echo "   • Build Command: pip install -r requirements.txt"
    echo "   • Start Command: gunicorn app:app"
    echo "   • Plan: Free"
    echo "6. 🚀 Click 'Create Web Service'"
    echo ""
    echo -e "${YELLOW}📋 Repository Information:${NC}"
    echo "• Repository URL: $REPO_URL"
    echo "• Branch: main"
    echo "• Auto-deploy: Enabled (recommended)"
    echo ""
    echo -e "${BLUE}🔗 View your repository: https://github.com/$GITHUB_USERNAME/$PROJECT_NAME${NC}"
    echo ""
    echo -e "${GREEN}📱 Once deployed, update link_to_my_web_app.txt with your live URL!${NC}"

else
    echo ""
    echo -e "${RED}❌ Failed to push to GitHub${NC}"
    echo ""
    echo -e "${YELLOW}💡 Troubleshooting:${NC}"
    echo "1. Make sure the repository exists on GitHub:"
    echo "   https://github.com/$GITHUB_USERNAME/$PROJECT_NAME"
    echo "2. Check your GitHub credentials"
    echo "3. Verify you have push access to the repository"
    echo ""
    echo -e "${BLUE}🔧 Manual steps if needed:${NC}"
    echo "1. Create repository on GitHub: https://github.com/new"
    echo "2. Repository name: $PROJECT_NAME"
    echo "3. Make it public"
    echo "4. Don't initialize with README"
    echo "5. Run this script again"

    exit 1
fi

echo ""
echo -e "${BLUE}🎯 Deployment checklist:${NC}"
echo "✅ Git repository initialized"
echo "✅ Code committed to Git"
echo "✅ Pushed to GitHub"
echo "⏳ Next: Deploy on Render"
echo "⏳ Next: Update hosting link file"
echo ""
echo -e "${GREEN}Happy deploying! 🚀${NC}"
