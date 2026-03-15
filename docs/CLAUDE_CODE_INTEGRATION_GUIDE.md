# Claude Code Integration Guide

This document is the implementation handoff for integrating the current jersey detection API into a Next.js application through a server-side proxy route.

## Current API Behavior

- Endpoint: `POST /detect`
- Processing model: synchronous request/response
- Response on success: `200` with a JSON array of detections
- Response when no jersey is found: `200 []`
- There is no job ID, queue endpoint, or polling endpoint in the current API
- Recommendation: call the API only from a server-side Next.js route, not directly from the browser

## Health Endpoints

Use these before live traffic or during deployment checks:

- `GET /live` -> `200 {"status":"ok"}`
- `GET /ready` -> `200 {"status":"ok"}` when models and runtime dependencies are ready, otherwise `503`
- `GET /health` -> same behavior as `GET /ready`

Example not-ready response:

```json
{
  "status": "error",
  "detail": "Missing required runtime dependencies: ffmpeg (ffmpeg)"
}
```

## Request Schema

The API accepts either camelCase or snake_case for the aliased fields below. For consistency in a Next.js app, prefer camelCase.

### Required business fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `jerseyNumber` / `jersey_number` | integer | yes | Must be between `0` and `99` |
| `jerseyColor` / `jersey_color` | string | yes | Non-empty string |
| `sport` | string | yes | Must be one of `basketball`, `football`, `lacrosse` |

### Video source fields

Provide exactly one source in client code, even though validation only requires at least one.

| Field | Type | Required | Notes |
|---|---|---:|---|
| `videoUrl` / `video_url` | string | conditional | Public HTTP(S) URL such as Cloudinary or YouTube |
| `videoPath` / `video_path` | string | conditional | Local/server path, mainly for backend use |
| `videoBytesB64` / `video_bytes_b64` | string | conditional | Base64-encoded video bytes |

### Optional field

| Field | Type | Required | Notes |
|---|---|---:|---|
| `position` | string | no | Any non-empty string is accepted; current tuned priors are `guard`, `quarterback`, `midfielder` |

## Valid Values

### `sport`

Accepted values are case-insensitive in practice because the pipeline normalizes them before inference:

- `basketball`
- `football`
- `lacrosse`

Recommended: send lowercase values exactly as above.

### `position`

The field is optional. If provided, it is normalized to lowercase before scoring. The API currently has explicit position priors for:

- `basketball`: `guard`
- `football`: `quarterback`
- `lacrosse`: `midfielder`

Other non-empty strings will not fail validation, but they will not benefit from those tuned priors.

### `jerseyColor`

Validation requires only a non-empty string, but detection quality is best when you send a clear common color name or a six-digit hex color.

Best-supported named colors from the current HSV rules:

- `white`
- `black`
- `grey`
- `gray`
- `red`
- `scarlet`
- `maroon`
- `orange`
- `yellow`
- `gold`
- `green`
- `teal`
- `cyan`
- `blue`
- `royal blue`
- `navy`
- `purple`
- `violet`
- `pink`
- `magenta`

Also supported:

- CSS/web named colors that can be resolved by `webcolors`
- Hex RGB strings such as `#0000FF`, `#FFFFFF`, `#FFD700`

Recommended: send either a value from the named list above or a precise `#RRGGBB` value.

## Example Request

```json
{
  "videoUrl": "https://res.cloudinary.com/your-cloud/video/upload/v1234567890/game.mp4",
  "jerseyNumber": 10,
  "jerseyColor": "blue",
  "sport": "basketball",
  "position": "guard"
}
```

Equivalent snake_case request:

```json
{
  "video_url": "https://res.cloudinary.com/your-cloud/video/upload/v1234567890/game.mp4",
  "jersey_number": 10,
  "jersey_color": "blue",
  "sport": "basketball",
  "position": "guard"
}
```

## Response Schema

On success the API returns an array of detections. Each array element has this shape:

```json
{
  "timestamp": 8.4,
  "confidence": 0.92,
  "bbox": {
    "x1": 340,
    "y1": 180,
    "x2": 490,
    "y2": 520,
    "x1_pct": 21.25,
    "y1_pct": 15.0,
    "x2_pct": 30.63,
    "y2_pct": 43.33
  }
}
```

Full example response:

```json
[
  {
    "timestamp": 8.4,
    "confidence": 0.92,
    "bbox": {
      "x1": 340,
      "y1": 180,
      "x2": 490,
      "y2": 520,
      "x1_pct": 21.25,
      "y1_pct": 15.0,
      "x2_pct": 30.63,
      "y2_pct": 43.33
    }
  },
  {
    "timestamp": 9.1,
    "confidence": 0.88,
    "bbox": {
      "x1": 338,
      "y1": 178,
      "x2": 488,
      "y2": 518,
      "x1_pct": 21.13,
      "y1_pct": 14.83,
      "x2_pct": 30.5,
      "y2_pct": 43.17
    }
  }
]
```

## Important Contract Notes

- The current API returns frame-level detections, not clip segments
- The current API does not return `start_time`, `end_time`, `primary_action`, or `player_position` in the response
- `timestamp` is the sampled frame timestamp in seconds
- `confidence` is a normalized detection confidence between `0.0` and `1.0`
- `bbox` contains both pixel coordinates and percentage coordinates

## Error Cases

### `200 OK` with detections

The target jersey was found in one or more frames.

### `200 OK` with an empty array

```json
[]
```

Meaning:

- The request was valid
- The pipeline ran
- No final detections passed filtering for the requested jersey

Treat this as a valid "no result" outcome, not as a transport failure.

### `400 Bad Request`

Returned when the JSON body is invalid or required fields fail validation.

Example:

```json
{
  "error": "body -> jerseyNumber: Value error, 'jersey_number' must be between 0 and 99."
}
```

Common causes:

- missing video source
- missing `jerseyNumber`
- missing `jerseyColor`
- missing `sport`
- invalid `sport`
- invalid base64 in `videoBytesB64`

### `503 Service Unavailable`

Returned when the app started but the detector has not finished warming up or failed startup checks.

Example:

```json
{
  "error": "Detection service is not ready: Missing required runtime dependencies: yt-dlp (yt-dlp)"
}
```

### `500 Internal Server Error`

Returned when something fails during request processing, such as download, decode, or model execution.

Example:

```json
{
  "error": "Internal detection error. See server logs for details."
}
```

Important: video download failures currently surface as `500` with the generic error above. The detailed reason is in the server logs, not in the response body.

## Timeout Guidance

The current service is synchronous. A single request may stay open for a long time on CPU, especially for longer videos.

Recommended settings:

- Next.js proxy fetch timeout: `180s` to `300s`
- Upstream server timeout: at least `1800s` for deployment parity with the current Docker setup
- Browser timeout: do not call the detection service directly from the browser

Implementation guidance:

- If the clip is short and the request originates from a server-side route, waiting for a single response is acceptable
- If you want user-visible progress, wrap this API in your own job table and poll your own app, not the detection API directly
- Do not implement polling against `/detect` itself because it is not a queued job API

## Authentication and Headers

Current API behavior:

- No API key header is required by the FastAPI service
- `Content-Type: application/json` should be sent
- No special response headers are required for parsing

Recommended deployment pattern:

- Keep the FastAPI URL private
- Put a server-side Next.js route in front of it
- Add authentication and rate limiting on your application side if needed

## BBox Percentage Mapping

The percentage values are percentages of the full frame dimensions, in the `0..100` range.

Interpretation:

- `x1_pct` and `x2_pct` are percentages of frame width
- `y1_pct` and `y2_pct` are percentages of frame height

To compute the spotlight center:

```ts
const centerXPct = (bbox.x1_pct + bbox.x2_pct) / 2;
const centerYPct = (bbox.y1_pct + bbox.y2_pct) / 2;
```

To convert back to pixels for a rendered frame:

```ts
const x1 = (bbox.x1_pct / 100) * frameWidth;
const y1 = (bbox.y1_pct / 100) * frameHeight;
const x2 = (bbox.x2_pct / 100) * frameWidth;
const y2 = (bbox.y2_pct / 100) * frameHeight;
```

To size a spotlight circle from the bbox:

```ts
const widthPx = x2 - x1;
const heightPx = y2 - y1;
const radiusPx = Math.max(widthPx, heightPx) * 0.6;
```

## Recommended Next.js Proxy Route

Use the Node.js runtime, not the Edge runtime, because the request can be long-running.

```ts
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const UPSTREAM_URL = process.env.JERSEY_API_BASE_URL;

export async function POST(request: NextRequest) {
  if (!UPSTREAM_URL) {
    return NextResponse.json(
      { error: "Missing JERSEY_API_BASE_URL" },
      { status: 500 }
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "Invalid JSON body" },
      { status: 400 }
    );
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 180_000);

  try {
    const upstream = await fetch(`${UPSTREAM_URL}/detect`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal: controller.signal,
      cache: "no-store",
    });

    const text = await upstream.text();
    let data: unknown = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = { error: "Upstream returned a non-JSON response", raw: text };
    }

    return NextResponse.json(data, {
      status: upstream.status,
      headers: {
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    const isAbort =
      error instanceof Error &&
      (error.name === "AbortError" || error.message.includes("aborted"));

    return NextResponse.json(
      {
        error: isAbort
          ? "Detection request timed out while waiting for the upstream API"
          : "Failed to reach detection API",
      },
      { status: 504 }
    );
  } finally {
    clearTimeout(timeout);
  }
}
```

## Integration Checklist

- Use camelCase in the Next.js client and proxy route
- Send exactly one video source field
- Treat `200 []` as a valid no-detection result
- Treat `503` as a deployment/readiness issue
- Treat `500` as an upstream processing failure and check server logs
- Use a server-side route with a long timeout
- Use bbox percentage coordinates for overlay placement
- Do not build polling against `/detect` until the API exposes async job semantics
