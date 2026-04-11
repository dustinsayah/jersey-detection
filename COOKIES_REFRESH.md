# Refreshing YouTube Cookies

Takes 3 minutes every 3-5 days. No redeploy needed.

## Steps

1. Go to [youtube.com](https://www.youtube.com) in Chrome
2. Make sure you're logged into YouTube Premium
3. Click the "Get cookies.txt LOCALLY" extension icon
4. Click Export and save as `cookies.txt`
5. Upload to Railway:
   ```bash
   curl -X POST https://jersey-detection-production-d8d8.up.railway.app/upload-cookies \
     --data-binary @cookies.txt
   ```
6. Verify:
   ```bash
   curl https://jersey-detection-production-d8d8.up.railway.app/health | jq '.cookie_health'
   ```

## How to Check Status

```bash
curl -s https://jersey-detection-production-d8d8.up.railway.app/health | jq '.cookie_health.estimated_days_remaining'
```

- 2+ days: Good
- 1-2 days: Plan refresh soon
- <1 day: Refresh now (warning appears in API responses)
- 0 days: Expired (downloads will fall back to non-cookie strategies)

## Notes

- Cookies are used as a FALLBACK strategy. WARP proxy is the primary download method.
- Cookie upload replaces the file on the running container (no rebuild needed).
- Cookies persist until the next Railway deploy (then need re-upload).
- The `/health` endpoint shows `cookie_health.estimated_days_remaining`.
- When cookies are <2 days from expiry, the `/analyze` response includes a `cookie_warning` field.
