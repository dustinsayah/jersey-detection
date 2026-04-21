# Setting Up Clipt Home Proxy on a Dedicated Computer

## What This Does

Keeps YouTube downloads working 24/7 without you needing to be at your main computer.
The home proxy downloads YouTube videos from a residential IP (which YouTube trusts more than cloud servers),
then streams them to the Railway backend for AI analysis.

Traffic flow: `cliptapp.com → Railway → proxy.cliptapp.com → this computer → YouTube`

## Requirements

- Windows 10 or 11
- Always-on internet connection (residential, not VPN)
- Python 3.10+
- ~2 GB free disk space

## One-Time Setup (10 minutes)

### Step 1: Install Python

Download from https://python.org if not installed.
**Important:** Check "Add Python to PATH" during installation.

Verify:
```
python --version
```

### Step 2: Copy Files

Copy the entire `scripts/` folder to `C:\clipt-proxy\`

You need these files:
- `home_proxy.py` — the Flask server
- `update_railway.py` — auto-updates Railway when tunnel URL changes
- `second_computer_startup.bat` — startup script (use this one)
- `.tunnel-token` — Cloudflare tunnel authentication
- `.railway-token` — Railway API token for auto-updating (ask Dustin for this)

### Step 3: Install Dependencies

Open Command Prompt and run:
```
cd C:\clipt-proxy
pip install flask yt-dlp requests
```

### Step 4: Verify the Tunnel Token

The file `C:\clipt-proxy\.tunnel-token` should already exist from Step 2.
Verify it has content:
```
type .tunnel-token
```
It should show a long string starting with `eyJ...`

### Step 5: Install cloudflared

Option A (winget):
```
winget install --id Cloudflare.cloudflared
```

Option B (manual):
Download from https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
Save to `C:\Users\YourName\AppData\Local\cloudflared.exe`

Verify:
```
cloudflared --version
```

### Step 6: Test It

Double-click `second_computer_startup.bat`

You should see:
```
CLIPT HOME PROXY - Starting...
Home proxy is running on port 5050
Using PERMANENT Cloudflare tunnel...
URL: proxy.cliptapp.com (never changes)
```

Verify it's working by checking:
https://jersey-detection-production-d8d8.up.railway.app/health

Look for: `"home_proxy": {"status": "online"}`

### Step 7: Auto-Start on Boot

1. Right-click `second_computer_startup.bat` → **Send to** → **Desktop (create shortcut)**
2. Right-click the shortcut → **Properties** → **Shortcut** tab → **Run:** select **Minimized**
3. Press `Win+R`, type `shell:startup`, press Enter
4. Move the shortcut into that Startup folder

This makes the proxy start automatically when the computer boots.
When the tunnel URL changes (on restart), Railway is updated automatically — no manual steps.

## Daily Operation

- Computer must stay **on** and **connected to internet**
- `run_home_proxy.bat` runs automatically on startup (after Step 7)
- Green indicator on cliptapp.com/setup-home-proxy means it's working
- The proxy uses minimal CPU/RAM — fine to run alongside normal computer use

## Monitoring

### Quick Check
Visit: https://jersey-detection-production-d8d8.up.railway.app/health

Look for:
```json
"home_proxy": {
  "status": "online",
  "url": "https://proxy.cliptapp.com"
}
```

### Dashboard
Visit: https://cliptapp.com/setup-home-proxy
The status banner shows green when the proxy is connected.

## Troubleshooting

### "Home proxy failed to start on port 5050"
Another program is using port 5050. Close it or change the port in `home_proxy.py`.

### "cloudflared not found"
Re-install cloudflared (Step 5). Make sure it's in your PATH.

### Health shows "offline"
1. Is the computer on and connected to internet?
2. Is `run_home_proxy.bat` running? (Check taskbar for "Clipt Home Proxy" window)
3. Try restarting: close all Clipt windows and double-click `run_home_proxy.bat`

### YouTube downloads fail
YouTube cookies may need refreshing. Visit cliptapp.com/setup-home-proxy for cookie upload instructions.

## Maintenance

- **Monthly:** Restart the computer to refresh connections
- **When cookies expire:** Upload fresh cookies via cliptapp.com/setup-home-proxy
- **yt-dlp updates:** Run `pip install --upgrade yt-dlp` monthly
