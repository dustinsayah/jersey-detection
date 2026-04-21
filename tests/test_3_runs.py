"""Run 3 consecutive detection tests against Railway."""
import requests
import time
import json
import sys

RAILWAY = "https://jersey-detection-production-d8d8.up.railway.app"

# Health check
h = requests.get(f"{RAILWAY}/health", timeout=10).json()
print(f"Health: v{h['version']}, home_proxy={h['home_proxy']['status']}, "
      f"cookies={h.get('cookie_health', {}).get('estimated_days_remaining', '?')} days")

payload = {
    "videoUrl": "https://www.youtube.com/watch?v=BKorP55Aqvg",
    "jersey_number": "2",
    "sport": "football",
    "position": "quarterback",
    "jersey_color": "navy",
    "startTime": 0,
    "endTime": 600,
}

results = []
for run in range(3):
    print(f"\n=== RUN {run+1}/3 ===")
    try:
        resp = requests.post(f"{RAILWAY}/analyze-async", json=payload, timeout=30)
        rj = resp.json()
        job_id = rj.get("job_id")
        if not job_id:
            print(f"No job_id: {rj}")
            results.append({"run": run+1, "pass": False, "error": f"no job_id: {json.dumps(rj)[:200]}"})
            continue
        print(f"Job: {job_id}")

        start = time.time()
        while time.time() - start < 900:
            time.sleep(15)
            try:
                poll = requests.get(f"{RAILWAY}/analyze-jobs/{job_id}", timeout=10).json()
            except Exception as pe:
                print(f"[{int(time.time()-start)}s] poll error: {pe}")
                continue
            status = poll.get("status", "unknown")
            progress = poll.get("progress", 0)
            message = poll.get("message", "")
            elapsed = int(time.time()-start)
            print(f"[{elapsed}s] {status} {progress}% {message[:80]}")

            if status == "complete":
                result = poll.get("result", {})
                clips = result.get("detections", result.get("clips", []))
                strategy = result.get("download_strategy", result.get("strategy_used", "unknown"))
                dl_time = result.get("download_time_s", "?")
                video_h = result.get("video_height", "?")
                print(f"PASS: {len(clips)} clips, strategy={strategy}, height={video_h}, dl_time={dl_time}s")
                if clips:
                    c = clips[0]
                    print(f"First clip: start={c.get('start_time','?')}, end={c.get('end_time','?')}, "
                          f"type={c.get('play_type','?')}, jersey={c.get('jerseyNumberSeen', c.get('jersey_number','?'))}")
                results.append({
                    "run": run+1, "pass": True, "clips": len(clips),
                    "strategy": strategy, "time": elapsed,
                    "height": video_h, "dl_time": dl_time,
                })
                break
            elif status == "failed":
                error = poll.get("error", "unknown")
                print(f"FAIL: {error[:300]}")
                results.append({"run": run+1, "pass": False, "error": error[:200], "time": elapsed})
                break
        else:
            print("TIMEOUT after 900s")
            results.append({"run": run+1, "pass": False, "error": "timeout", "time": 900})
    except Exception as e:
        print(f"Exception: {e}")
        results.append({"run": run+1, "pass": False, "error": str(e)[:200]})

    if run < 2:
        print("Waiting 30s before next run...")
        time.sleep(30)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
for r in results:
    if r.get("pass"):
        print(f"Run {r['run']}: PASS - {r['clips']} clips via {r['strategy']} in {r['time']}s (height={r.get('height','?')})")
    else:
        print(f"Run {r['run']}: FAIL - {r.get('error', 'unknown')}")

all_passed = all(r.get("pass") and r.get("clips", 0) > 0 for r in results)
print(f"\n{'ALL 3 PASSED' if all_passed else 'SOME FAILED'} - {len([r for r in results if r.get('pass')])} of 3 passed")
