# Deploying Bungii to Netlify

This document provides instructions for deploying the Bungii frontend to Netlify.

## Prerequisites

- A Netlify account
- Git repository with your Bungii codebase
- A deployed backend API (the frontend will connect to this)

## Deployment Steps

1. **Login to Netlify**
   - Go to [netlify.com](https://netlify.com) and sign in or create an account

2. **Create a New Site**
   - Click "Add new site" → "Import an existing project"
   - Connect to your Git provider and select your Bungii repository

3. **Configure Build Settings**
   - Set the following build configuration:
     - Base directory: `auditpulse_mvp/frontend`
     - Build command: `npm run build`
     - Publish directory: `dist`

4. **Configure Environment Variables**
   - In your site settings, go to "Build & deploy" → "Environment"
   - Add these variables:
     - `VITE_API_URL`: URL to your backend API (e.g., https://api.bungii.com)

5. **Update API URL**
   - Edit the `/api/*` redirect in `netlify.toml` and `_redirects` to point to your actual backend API
   - Change `https://bungii-api.your-backend-url.com` to your actual API domain

6. **Deploy**
   - Trigger a deploy from the Netlify dashboard
   - Netlify will build and deploy your frontend application

## Custom Domain Setup

1. Go to "Domain settings" in your Netlify site dashboard
2. Click "Add custom domain"
3. Follow the instructions to configure DNS settings for your domain

## Troubleshooting

- If assets are not loading, check the paths in your code to ensure they're relative
- If API calls fail, verify the redirect rules in `_redirects` and `netlify.toml` 
- Check build logs for any errors during the deployment process

## Backend Deployment

Note that the backend API needs to be deployed separately. Options include:
- AWS Elastic Beanstalk
- Heroku
- Google Cloud Run
- DigitalOcean App Platform 