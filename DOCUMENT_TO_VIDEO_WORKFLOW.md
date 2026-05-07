# `POST /gemini/document-to-video` — End-to-End Workflow

This document explains exactly what happens when a client uploads a document to
`POST /gemini/document-to-video` and walks through every stage of the pipeline,
the on-disk artifacts it produces, the retry / resume / partial-recovery
behaviour, and the operational rules for running long jobs.

The endpoint implements the duration-aware pipeline described in
[`APPROACH.md`](./APPROACH.md). Its goal is to take a single uploaded document
and return one final stitched video whose visuals never break mid-narration,
even when individual scenes need more than 8 seconds of screen time.

---

## 1. Endpoint contract

### Route

```
POST /gemini/document-to-video
Content-Type: multipart/form-data
```

### Request

| Field | Where | Type | Required | Notes |
|---|---|---|---|---|
| `file` | multipart form field | binary | yes | The source document. Max 15 MB. Supported: PDF, DOCX, plain text formats (TXT, MD, CSV, JSON, XML, HTML). Legacy `.doc` is rejected. |
| `aspectRatio` | query string | `"9:16"` \| `"16:9"` | no | Defaults to `9:16`. Anything other than `16:9` is normalized to `9:16`. |
| `resumeMontageId` | query string | string | no | Reuse the workspace folder of a previous run instead of creating a new one. See §7. |
| `resumeFromScene` | query string | integer ≥ 1 | no | 1-based scene index to resume from. Defaults to 1. Errors out if greater than the planned scene count. |

### Success response (HTTP 201)

```json
{
  "originalFileName": "owner-manual.pdf",
  "extractedTextLength": 184312,
  "summaryTextLength": 9241,
  "summary": "...",
  "aspectRatio": "9:16",
  "montage": {
    "combinedFilePath": "<abs path>/generated-document-to-video/montage_dur_<ts>.mp4",
    "combinedFileName": "montage_dur_<ts>.mp4",
    "montageId": "montage_dur_<ts>",
    "montageDir": "<abs path>/generated-document-to-video/montage_dur_<ts>",
    "imagesDir": "<...>/images",
    "clipsDir": "<...>/clips",
    "scenesDir": "<...>/scenes",
    "scenes": [
      {
        "id": "scene_01",
        "sceneIndex": 0,
        "description": "Wide establishing shot of...",
        "script": "Welcome to the cell biology lab...",
        "audioDurationSeconds": 18,
        "clipsNeeded": 3,
        "referenceImages": ["<...>/images/scene_01_frame_01.png", "..."]
      }
    ],
    "scenesStitched": [
      { "sceneId": "scene_01", "sceneIndex": 0, "videoPath": "<...>/scenes/scene_01.mp4", "trimmedDurationSeconds": 18 }
    ],
    "clipFiles": ["<...>/clips/scene_01/clip_01.mp4", "..."],
    "imagePaths": ["<...>/images/scene_01_frame_01.png", "..."],
    "styleBrief": "Color palette: ...",
    "totalDurationSeconds": 312
  }
}
```

### Error responses

| HTTP status | When | Shape |
|---|---|---|
| `400 Bad Request` | Missing `file`, unsupported file type, or invalid `resumeFromScene`. | `{ error }` |
| `502 Bad Gateway` | Pipeline failed after retries (Veo / OpenAI / FFmpeg). The body includes `failureDetails` with the partial-recovery output (see §7) so the client can resume or download the already-rendered prefix. | `{ error, message, failureDetails }` |
| `500 Internal Server Error` | Anything else (programming error, unexpected exception). | `{ error }` |

---

## 2. Pipeline at a glance

```
file
  │
  ▼
extract plain text (PDF / DOCX / text)
  │
  ▼
OpenAI summary (single instructional summary covering whole document)
  │
  ▼
scene + narration plan ─►  scenes.json (description + script per scene)
  │                          ⮡ derive audioDurationSeconds (~2.5 words/sec)
  │                          ⮡ derive clipsNeeded = ceil(audio / 8)
  ▼
Gemini "visual style bible" (shared across every keyframe)
  │
  ▼
per scene
  ├── generate N keyframe images (start / mid-i / end)
  └── per clip in scene
        ├── pick start image: previous clip's last frame ── else scene's start frame
        ├── Veo image-to-video → 8 s clip
        └── extract last frame → seed for next clip
  ▼
per scene: ffmpeg concat N clips → trim to audioDurationSeconds → scenes/scene_NN.mp4
  ▼
ffmpeg concat all scene videos → generated-document-to-video/<montageId>.mp4
```

---

## 3. Step-by-step

### Step 0 — Request handling

`GeminiController.documentToVideo` (`src/gemini/gemini.controller.ts`)
validates the multipart payload, parses `aspectRatio`, and parses
`resumeMontageId` + `resumeFromScene`. It then delegates to
`GeminiService.generateDurationAwareVideoFromContent`.

If anything later throws a `SceneImageMontageError`, the controller responds
with `502 Bad Gateway` and includes `failureDetails` (partial-recovery output,
the step that failed, retry counts, file paths) so the client can resume.

### Step 1 — Text extraction

`DocumentTopicsService.extractPlainText`
(`src/document-topics/document-topics.service.ts`).

* PDF → `pdf-parse`
* DOCX → `mammoth`
* `text/*`, `.txt`, `.md`, `.csv`, `.json`, `.xml`, `.html`, `.rtf` → UTF-8 read
* `.doc` (legacy) → 400, with a "save as .docx or PDF" hint
* Anything else → 400

### Step 2 — Document summarization

`DocumentTopicsService.summarizePlainText` calls OpenAI Responses with a
configurable model (default `gpt-5`, falls back to `gpt-4.1`) and a system
prompt that demands:

* one paragraph per topic, in original document order
* full coverage (no skipping sections)
* plain prose, no JSON / markdown
* a length proportional to document depth (400–700 words for short, up to
  1500–3000+ words for long technical documents)

The summary is the input to every subsequent stage; if it is empty the
pipeline aborts with a 5xx.

### Step 3 — Preflight + workspace

Inside `generateDurationAwareVideoFromContent`:

1. `assertFfmpegAvailable()` runs `ffmpeg -version` once and caches the
   result. The pipeline only stitches at the very end, so without this check
   we'd spend Veo cost and crash on the final concat. If ffmpeg is missing,
   the run fails fast with a clear error before any LLM / Veo call.
2. The output root is fixed at `<cwd>/generated-document-to-video/`.
3. `montageId` is either the supplied `resumeMontageId` or
   `montage_dur_<timestamp>`.
4. The workspace is created:

   ```
   generated-document-to-video/<montageId>/
   ├── images/   (keyframe PNG/JPG per scene per frame)
   ├── clips/    (raw 8 s Veo clips, one subfolder per scene)
   └── scenes/   (concat + trimmed per-scene mp4s)
   ```

5. A copy-pasteable resume hint is logged immediately:

   ```
   [duration-aware-video] Resume hint: if this run is interrupted, re-POST
     /gemini/document-to-video?resumeMontageId=montage_dur_<ts> ...
   ```

   Save that line. If anything dies later you can resume cleanly (§7).

### Step 4 — Scene + narration plan

`geminiVideoScriptWithNarration(content)` calls OpenAI with a strict
JSON-array prompt and produces:

```json
[
  { "description": "Wide establishing shot of ...", "script": "Welcome to ..." },
  { "description": "Macro close-up of gloved hands ...", "script": "Carefully transfer ..." }
]
```

* `description` is the visual prompt (camera movement, lighting, environment,
  educational tone).
* `script` is the narrator voice-over text. **The script length is what
  drives clip count for the rest of the pipeline.**

The model is retried up to `VIDEO_SCRIPT_MAX_RETRIES = 3` times on JSON parse
errors and truncation symptoms.

### Step 5 — Duration-aware planning

`planScenes()` converts each `{description, script}` into a `PlannedScene`:

| Field | How it's computed |
|---|---|
| `id` | `scene_NN` (1-based, 2-digit padded) |
| `sceneIndex` | 0-based |
| `audioDurationSeconds` | `ceil(wordCount / 2.5)`, min 1 |
| `clipsNeeded` | `max(1, ceil(audioDurationSeconds / 8))` |
| `referenceImages` | filled in during Step 7 |

**This is the core fix vs. the old pipeline.** A 24-second scene now plans
`clipsNeeded = 3`; a 4-second scene plans `clipsNeeded = 1` and gets
trimmed back to 4 seconds at stitch time.

The planned scenes are persisted to `<montageDir>/scenes.json`. This file is
the source of truth for resume runs — see §7.

### Step 6 — Visual style brief

`geminiVisualStyleBrief(content, scenes.map(s => s.description))` produces a
single 120–200 word "visual bible" describing palette, lighting, recurring
characters, props, environment, and lens language. This brief is fed into
every keyframe prompt so all images share the same look.

### Step 7 — Per-scene keyframe images

For each scene, the pipeline generates exactly `clipsNeeded` reference
images, labelled by `frameLabelsForClipCount`:

| `clipsNeeded` | Frame labels |
|---|---|
| 1 | `establishing` |
| 2 | `start`, `end` |
| 3 | `start`, `mid-1`, `end` |
| N | `start`, `mid-1`, ..., `mid-(N-2)`, `end` |

Each keyframe is generated by `generateSceneKeyframeFlashImage` (Gemini
2.5 Flash Image). If Flash refuses or errors, the pipeline falls back per
frame to `generateSceneKeyframeImagen` (Imagen 3).

The very first generated frame (or the first frame found on disk during a
resume) becomes the **visual anchor** that's passed as a reference image to
every subsequent keyframe call, so character / wardrobe / props stay
consistent across the whole video.

Files land at:

```
images/scene_01_frame_01.png
images/scene_01_frame_02.png
images/scene_02_frame_01.png
...
```

If a file with the right name (`scene_NN_frame_FF.{png,jpg,jpeg}`) already
exists and is non-empty, generation for that frame is skipped — that's how
resume reuses prior work.

### Step 8 — Veo clips with last-frame chaining

For each scene, the pipeline iterates `c = 0 … clipsNeeded - 1` and:

1. Picks the **start image**:
   * `c = 0` → `scene.referenceImages[0]` (the `start` / `establishing` keyframe)
   * `c > 0` → the JPG extracted from the last frame of clip `c - 1`
2. Builds a continuity-aware prompt:
   * Clip 1 prompt: "Open this scene with cinematic motion that establishes
     the situation."
   * Clip 2+ prompt: "This is clip N of M for the same continuous scene.
     The provided image is the FINAL frame of the previous clip — continue
     the same shot without cutting, jumping, or changing subjects, wardrobe,
     props, environment, or camera angle. Pick up motion exactly where it
     left off."
3. Calls Veo (`veo-3.1-generate-preview`) at 720p, 1 video, with the
   selected start image as `image`.
4. Polls the long-running operation up to ~20 minutes.
5. Validates the response: explicit error checks for finished-with-error
   LROs and missing video files (with diagnostic snapshots in logs).
6. Downloads the mp4 to `clips/scene_NN/clip_CC.mp4`.
7. If this is not the last clip in the scene, `extractLastFrame` runs:

   ```
   ffmpeg -y -sseof -0.1 -i clip_CC.mp4 -frames:v 1 -q:v 2 clip_CC_lastframe.jpg
   ```

   That JPG seeds clip `CC + 1`.

Veo retries (`runMontageVeoClipWithRetries`) handle the most common
transient failures:

* gRPC `INTERNAL` (code 13), `UNAVAILABLE` (14), `OUT_OF_RANGE`-style
  payload errors, deadline / timeout, ECONNRESET, plain "try again" /
  "internal server" messages.
* Up to `VEO_MONTAGE_CLIP_MAX_ATTEMPTS = 5` attempts per clip with a linear
  backoff base of 25 s.

If any clip already exists on disk for the scene, it is reused; the
last-frame JPG is also re-extracted if it's missing, so chain continuity
survives a resume.

### Step 9 — Per-scene stitch + trim

After all clips for a scene are on disk, `stitchAndTrimSceneClips` produces
one `scenes/scene_NN.mp4`:

* **1 clip** → trim only (`ffmpeg -i clip_01.mp4 -t <audio> -c copy ...`)
  with a re-encode (`libx264 + aac`) fallback if stream copy refuses.
* **>1 clip** → write a concat list, run
  `ffmpeg -f concat -safe 0 -i list.txt -c copy ...`,
  fall back to `concatClipsFFmpegReencode` if stream copy fails (different
  encoding params), then trim the result to `audioDurationSeconds`. The
  intermediate `*_concat.mp4` is deleted afterwards.

Each scene video is therefore exactly the length of its narration estimate,
which is the whole point of the refactor: long scenes no longer get cut at
8 seconds, and short scenes don't pad with "extra" Veo content.

### Step 10 — Final concat

Once all `scenes/scene_NN.mp4` files exist, the pipeline concatenates them
into `<outDir>/<montageId>.mp4`:

* First try `concatClipsFFmpeg` (stream copy, fast).
* If that fails (different stream params after the per-scene re-encode
  fallback) it falls back to `concatClipsFFmpegReencode`
  (`-filter_complex concat`, `+faststart`).

Total duration of the response equals
`sum(scene.audioDurationSeconds for scene in scenes)`.

---

## 4. On-disk workspace layout

For a run that produces 3 scenes with 1, 3, and 2 clips respectively:

```
generated-document-to-video/
├── montage_dur_1777441095960/
│   ├── scenes.json                          # canonical plan (scenes + clipsNeeded)
│   ├── images/
│   │   ├── scene_01_frame_01.png            # scene 1 needs 1 clip → 1 frame
│   │   ├── scene_02_frame_01.png            # scene 2 needs 3 clips → 3 frames
│   │   ├── scene_02_frame_02.png
│   │   ├── scene_02_frame_03.png
│   │   ├── scene_03_frame_01.png            # scene 3 needs 2 clips → 2 frames
│   │   └── scene_03_frame_02.png
│   ├── clips/
│   │   ├── scene_01/
│   │   │   └── clip_01.mp4
│   │   ├── scene_02/
│   │   │   ├── clip_01.mp4
│   │   │   ├── clip_01_lastframe.jpg
│   │   │   ├── clip_02.mp4
│   │   │   ├── clip_02_lastframe.jpg
│   │   │   └── clip_03.mp4
│   │   └── scene_03/
│   │       ├── clip_01.mp4
│   │       ├── clip_01_lastframe.jpg
│   │       └── clip_02.mp4
│   └── scenes/
│       ├── scene_01.mp4
│       ├── scene_02.mp4
│       └── scene_03.mp4
└── montage_dur_1777441095960.mp4            # final output
```

Failed runs additionally produce
`montage_dur_<ts>_partial_<ts>.mp4` at the workspace root containing every
per-scene video that completed before the failure (see §7).

---

## 5. Models, retries, and configuration

### Model identifiers (in `gemini.service.ts`)

| Constant | Default | Purpose |
|---|---|---|
| `OPENAI_SCENE_MODEL` | `gpt-5` | Scene + narration plan. Override with env `OPENAI_SCENE_MODEL`. |
| `FLASH_IMAGE_MODEL` | `gemini-2.5-flash-image` | Primary keyframe generator. |
| `IMAGEN_GENERATE_MODEL` | `imagen-3.0-generate-002` | Per-frame fallback when Flash refuses or errors. |
| `VEO_MODEL` | `veo-3.1-generate-preview` | Image-to-video for every clip. |

Document summarization model lives in `DocumentTopicsService`:

| Env var | Default | Purpose |
|---|---|---|
| `OPENAI_SUMMARY_MODEL` | `gpt-5` | Summary of the whole document. |
| `OPENAI_SUMMARY_MODEL_FALLBACKS` | `gpt-4.1` | Comma-separated fallback list. |
| `OPENAI_TOPIC_MODEL` | `gpt-5` | Topic segmentation (used by other endpoints). |
| `OPENAI_TOPIC_MODEL_FALLBACKS` | `gpt-4.1` | Comma-separated fallback list. |

### Required environment variables

| Var | Purpose |
|---|---|
| `OPENAI_API_KEY` | Scene + narration generation, document summarization. |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Flash image, Imagen, Veo. |

### Retry constants

| Constant | Value | Behaviour |
|---|---|---|
| `VIDEO_SCRIPT_MAX_RETRIES` | 3 | Scene+narration JSON, with linear backoff. |
| `PIPELINE_STEP_MAX_RETRIES` | 5 | Wrapped around scene gen, style brief, and each keyframe. |
| `PIPELINE_STEP_RETRY_BASE_MS` | 2 000 | Linear backoff base for the wrapper. |
| `VEO_MONTAGE_CLIP_MAX_ATTEMPTS` | 5 | Per Veo clip. |
| `VEO_MONTAGE_CLIP_RETRY_BASE_MS` | 25 000 | Linear backoff base for Veo clips. |

### Pipeline constants

| Constant | Value |
|---|---|
| `VEO_CLIP_DURATION_SECONDS` | 8 |
| `NARRATION_WORDS_PER_SECOND` | 2.5 |

---

## 6. Error model (`SceneImageMontageError`)

Every recoverable failure inside the orchestrator is wrapped in
`SceneImageMontageError` with `failureDetails`:

```ts
{
  step: string;                        // exact phase that failed
  montageId: string;                   // workspace id (use this to resume)
  montageDir: string;                  // absolute workspace path
  imagesDir: string;
  retriesPerStep: number;              // PIPELINE_STEP_MAX_RETRIES
  successfulImageCount: number;
  successfulClipCount: number;         // count of stitched scenes that finished
  partialCombinedFilePath: string|null;
  partialCombinedFileName: string|null;
  originalErrorMessage: string;
}
```

The controller surfaces this as `502 Bad Gateway`:

```json
{
  "error": "Document to video failed while generating montage from summary.",
  "message": "[duration-aware-video] failed at video clip scene 7/12 clip 2/3 after retries: ...",
  "failureDetails": { "...as above..." }
}
```

The values to pay attention to in the failure body are:

* `montageId` — pass it back as `?resumeMontageId=<id>` to continue.
* `partialCombinedFilePath` — a stitched mp4 of every scene that completed
  before the failure, ready to play back even if you never resume.

---

## 7. Resume + partial recovery

### Why resume exists

A full document-to-video run does dozens of LLM calls, dozens of image
generations, and tens (or hundreds) of Veo clips. A single failed Veo
operation, an OpenAI rate-limit, an FFmpeg quirk on one clip, or an
operational restart can otherwise waste 30+ minutes of work.

Every long-lived artifact is written to a deterministic path inside the
workspace. The orchestrator checks for those paths before doing any work.

### How a resume run behaves

If the request includes `?resumeMontageId=<id>`:

1. The workspace `<id>` is reused (not recreated).
2. If `<id>/scenes.json` exists, it is loaded as the canonical plan instead
   of regenerating scenes from the LLM. This is critical because re-running
   `geminiVideoScriptWithNarration` would otherwise produce a slightly
   different scene set with different `clipsNeeded` values, breaking every
   on-disk file path.
3. For every scene `i` from `resumeFromScene` (default 1) onward:
   * **Keyframes:** any `scene_NN_frame_FF.{png,jpg,jpeg}` already on disk
     is reused; only missing frames are generated.
   * **Veo clips:** any `clips/scene_NN/clip_CC.mp4` with non-zero size is
     reused; if its `_lastframe.jpg` is missing it is re-extracted so the
     chain stays intact for the next clip.
   * **Per-scene stitched video:** any `scenes/scene_NN.mp4` with non-zero
     size is reused as-is.
4. The final concat runs over all `scenes/scene_NN.mp4` once everything is
   in place.

For scenes with index `< resumeFromScene` the pipeline expects every file
to already exist; if a keyframe or clip is missing in those scenes, the run
fails with a clear "Resume requested from scene N but missing X" error so
you can either lower `resumeFromScene` or re-run from 1.

### Partial recovery on failure

Whenever the orchestrator's outer `try` catches an error,
`stitchSuccessfulScenes` runs. It concatenates every per-scene video that
already finished into a single playable mp4 at:

```
generated-document-to-video/<montageId>_partial_<ts>.mp4
```

The path is included in `failureDetails.partialCombinedFilePath` on the
`502` response, so a client can surface it as "we got 7 of 12 scenes,
here's the partial video, click here to resume".

### Resume cheat sheet

```bash
# A run died with montageId = montage_dur_1777441095960
# All progress is on disk under generated-document-to-video/montage_dur_1777441095960/

# Resume from where it stopped (auto-detected per file existence):
curl -X POST "http://localhost:3000/gemini/document-to-video?resumeMontageId=montage_dur_1777441095960" \
     -F "file=@./owner-manual.pdf"

# Resume forcibly from scene 14 onward (scenes 1..13 must already be on disk):
curl -X POST "http://localhost:3000/gemini/document-to-video?resumeMontageId=montage_dur_1777441095960&resumeFromScene=14" \
     -F "file=@./owner-manual.pdf"
```

The same document file should be passed because it still drives the
summarization step on the very first invocation — but on resume runs that
have a saved `scenes.json`, the summary is no longer used to regenerate the
scene plan.

---

## 8. Operational rules for long jobs

A typical document-to-video run for a 20-page manual lasts 20–60 minutes.
The HTTP request is held open for the full duration, which means the
process and the connection both need to stay alive. The repo is configured
for that, but only if you start the server correctly.

### Do

* Build first: `npm run build`.
* Run with a non-watch script and a 4 GB heap:
  ```bash
  npm run start:long
  ```
  (`start:long` is `node --max-old-space-size=4096 dist/main` — see
  `package.json`.)
* Save the `Resume hint` line that the pipeline prints right after the
  workspace is created. If anything goes wrong, that line tells you the
  exact resume URL to call.

### Don't

* **Don't run `npm run start:dev`** for real document-to-video jobs. Watch
  mode can restart the dev server mid-flight (e.g. on a stray editor save
  or a `.tsbuildinfo` write), which kills the in-flight request and your
  client receives `ECONNREFUSED`. The pipeline's outer `try/catch` cannot
  run when the OS process itself dies, so partial recovery won't run
  either.
* Don't add the workspace directories (`generated/`,
  `generated-document-to-video/`) to anything that watches files — they
  are already in `.gitignore` and `tsconfig.exclude` for that reason.
* Don't reuse a `montageId` across two semantically different documents.
  The `scenes.json` plan is keyed by montage id, not by document hash.

### Process safety net

`src/main.ts` registers `process.on('uncaughtException')` and
`process.on('unhandledRejection')` handlers that **log instead of letting
Node terminate the server**. This means a leaked promise rejection from
deep inside an SDK no longer kills the in-flight pipeline — the orchestrator
gets a chance to either complete normally or fail through its own
`SceneImageMontageError` path so the client receives an HTTP error and can
resume.

If you ever see a line beginning with `[FATAL] uncaughtException` or
`[FATAL] unhandledRejection`, please grab the stack — it points at the
exact site that needs an explicit `try/catch`.

### FFmpeg

FFmpeg must be on `PATH` of the shell that starts the server.
`assertFfmpegAvailable()` runs `ffmpeg -version` once and refuses to start
the pipeline if it can't find FFmpeg, so this fails loudly and cheaply
instead of after every Veo call has already cost money.

---

## 9. Companion endpoint: `POST /gemini/estimate-duration`

Same input as `/document-to-video`, but it stops after the scene+narration plan
— no images, no Veo, no FFmpeg. Use it to show the user what a full run will
produce before committing to the long, expensive job.

Both endpoints share the same `GeminiService.planDurationAwareScenesFromContent`
internally, so the numbers the estimate returns match the actual pipeline 1:1.

### Response

```json
{
  "originalFileName": "owner-manual.pdf",
  "extractedTextLength": 184312,
  "summaryTextLength": 9241,
  "summary": "...",
  "aspectRatio": "9:16",

  "totalScenes": 12,
  "totalClipsNeeded": 31,
  "clipDurationSeconds": 8,

  "totalEstimatedDurationSeconds": 246,
  "totalEstimatedDurationFormatted": "4m 6s",

  "totalRawVeoBudgetSeconds": 248,
  "totalRawVeoBudgetFormatted": "4m 8s",

  "scenes": [
    {
      "id": "scene_01",
      "sceneIndex": 0,
      "description": "Wide establishing shot of ...",
      "script": "Welcome to the cell biology lab. Today we will ...",
      "audioDurationSeconds": 22,
      "clipsNeeded": 3,
      "referenceImages": []
    }
  ]
}
```

### Field meanings

| Field | What it tells you |
|---|---|
| `totalScenes` | Number of planned scenes the real pipeline will produce. |
| `totalClipsNeeded` | Sum of `scene.clipsNeeded` across all scenes — i.e. how many Veo clips will actually be generated. |
| `totalEstimatedDurationSeconds` | Length of the **final stitched output** (sum of trimmed per-scene durations). This is what the viewer ends up watching. |
| `totalRawVeoBudgetSeconds` | `totalClipsNeeded × 8` — the **pre-trim Veo budget**. Use this for cost / render-time estimates because Veo bills and renders every clip as a full 8 seconds even when the per-scene trim later shortens it. |
| `scenes[i].audioDurationSeconds` | Trimmed length of scene `i` in the final video. |
| `scenes[i].clipsNeeded` | How many 8 s Veo clips scene `i` needs. |
| `scenes[i].referenceImages` | Always `[]` here; only populated during the real `/document-to-video` run. |

`totalRawVeoBudgetSeconds` will always be `≥ totalEstimatedDurationSeconds`
because every scene rounds its clip count up to the next multiple of 8 s. The
difference is the per-scene trim that the real pipeline applies before the
final concat.

---

## 10. Quick reference

* **Endpoint:** `POST /gemini/document-to-video`
* **Estimate endpoint (same input, plan only):** `POST /gemini/estimate-duration`
* **Controller:** `GeminiController.documentToVideo` /
  `GeminiController.estimateDuration` in `src/gemini/gemini.controller.ts`
* **Service entrypoints:**
  * Run: `GeminiService.generateDurationAwareVideoFromContent`
  * Plan-only: `GeminiService.planDurationAwareScenesFromContent`
  * (both in `src/gemini/gemini.service.ts`)
* **Workspace root:** `<cwd>/generated-document-to-video/`
* **Final output:** `<workspace root>/<montageId>.mp4`
* **Plan file:** `<workspace root>/<montageId>/scenes.json`
* **Resume URL:**
  `POST /gemini/document-to-video?resumeMontageId=<id>[&resumeFromScene=<N>]`
* **How to start the server for real runs:** `npm run build && npm run start:long`
