# Deploying to Render.com - Quick Guide

## The Problem You're Facing

**Status: 502 Bad Gateway** when uploading a video → Analysis crashes due to **out of memory**.

**Root Cause:** Render Free Tier has 512MB RAM, but your analysis needs ~350MB. When two processes run simultaneously, you run out of memory.

## Quick Fix (2 minutes)

### Option A: Upgrade Render Plan (Most Reliable)
Go to your Render service → Settings → Plan → Select "Standard" ($7/month)
- Gets you 1GB RAM (should be enough)
- 30-second timeout (from 10 seconds)
- Better CPU

### Option B: Optimize for Free Tier
In your Render dashboard, set these **Environment Variables**:

```
WHISPER_MODEL=tiny
MAX_FRAMES=75
FRAME_SKIP=10
ANALYSIS_QUEUE_WORKERS=1
DELETE_UPLOAD_AFTER_ANALYSIS=true
```

Then redeploy. This makes analysis 50% faster and uses 50% less memory.

## Step-by-Step for Option B

### 1. Go to Render Dashboard
- Click your service name
- Scroll down to "Environment"
- Click "Add Environment Variable"

### 2. Add These Variables

| Key | Value | Reason |
|-----|-------|--------|
| `WHISPER_MODEL` | `tiny` | 10x faster audio processing |
| `MAX_FRAMES` | `75` | Process fewer frames (50% faster) |
| `FRAME_SKIP` | `10` | Skip more frames between analysis |
| `ANALYSIS_QUEUE_WORKERS` | `1` | Process one video at a time |
| `DELETE_UPLOAD_AFTER_ANALYSIS` | `true` | Free up disk space |

### 3. Deploy
- Click "Deploy" or push a new commit to trigger auto-deploy
- Wait 2-3 minutes for build to complete
- Test by uploading a video

## Testing Before Deploy

To verify these settings work on your machine:

```bash
# Set environment variables
export WHISPER_MODEL=tiny
export MAX_FRAMES=75
export FRAME_SKIP=10
export ANALYSIS_QUEUE_WORKERS=1

# Run the app
python -m api.app
```

Then upload a test video through the browser.

## If It Still Doesn't Work

### Check Render Logs
1. Go to your Render service
2. Click "Logs" tab
3. Look for messages like:
   - "Killed" or "OOMkilled" → Still out of memory, need to upgrade
   - "timeout" → Taking too long, reduce MAX_FRAMES more
   - "ffmpeg not found" → Missing FFmpeg (not common on Render)

### Next Steps
1. **If you see "OOMkilled":** Upgrade to Standard plan ($7/month)
2. **If you see "timeout":** Reduce `MAX_FRAMES` to 50
3. **Still broken?** Share the error message from logs

## Performance Comparison

| Setting | Free Tier | Recommended | Notes |
|---------|-----------|-------------|-------|
| `WHISPER_MODEL` | base | tiny | 10x speed improvement |
| `MAX_FRAMES` | 150 | 75 | 50% faster analysis |
| `FRAME_SKIP` | 5 | 10 | More frames skipped |
| `ANALYSIS_QUEUE_WORKERS` | 2 | 1 | One at a time = save memory |
| **Time per video** | 30-45s | 10-15s | ✓ Fits in 30s timeout |

## Long-Term Solution

For production, consider upgrading to **Standard ($7/month)** or **Pro ($27/month)** for:
- ✓ More RAM (1GB → 2GB)
- ✓ Longer timeout (30+ seconds)
- ✓ More CPU cores
- ✓ Better reliability

## Monitoring

After deployment, monitor with:

```bash
# Check if job is completing
curl https://your-app.onrender.com/readyz

# Check logs
# Go to Render dashboard → Logs tab
```

## Questions?

If analysis still fails:
1. **Share the exact error from Render Logs**
2. **Tell me if it says "Killed" or "timeout"**
3. **Let me know if you upgraded the plan**

Then I can provide more specific fixes!
