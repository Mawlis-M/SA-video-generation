# Veo Video Generation Pipeline — Refactored Approach

## Overview

This document describes the updated NestJS pipeline for Veo-based video generation. The core problem with the previous approach was that **each scene always generated a single 8-second clip**, which was not enough to cover longer scene scripts or audio, causing mid-scene video breaks.

---

## Problem With Old Approach

```
Content → Scenes → 1 image per scene → Veo (8s clip) → Video
```

- Every scene produced exactly one 8-second clip regardless of audio/script length
- A 24-second scene script would be cut at 8 seconds
- No relationship between audio duration and number of clips generated
- No continuity between adjacent clips within the same scene

---

## New Approach

```
Content → Scenes (with duration) → N images per scene → N × Veo clips → Stitch → Trim to audio → Final video
```

### Key Changes

1. **Duration-aware scene planning** — each scene tracks its audio/script duration
2. **Dynamic clip count** — `clipsNeeded = Math.ceil(sceneDurationSeconds / 8)`
3. **Multi-frame image generation** — generate start, mid, and end reference images per scene for visual continuity
4. **Chained clip generation** — last frame of clip `i` is used as the reference image for clip `i+1`
5. **Stitch + trim** — concatenate N clips, then hard-trim to exact audio duration

---

## NestJS Service Structure

### Module Layout

```
video/
├── video.module.ts
├── video.service.ts              # orchestrator
├── scene/
│   ├── scene-planner.service.ts  # breaks content into scenes with duration
│   └── scene.interface.ts
├── image/
│   └── frame-generator.service.ts # generates N reference images per scene
├── veo/
│   └── veo.service.ts             # wraps Veo API, generates single 8s clip
├── stitch/
│   └── stitch.service.ts          # ffmpeg concat + trim logic
└── dto/
    └── generate-video.dto.ts
```

---

## Interfaces

```typescript
// scene/scene.interface.ts

export interface Scene {
  id: string;
  sceneIndex: number;
  description: string;         // visual description for image generation
  script: string;              // narration / dialogue
  audioDurationSeconds: number; // measured or estimated audio length
  clipsNeeded: number;         // ceil(audioDurationSeconds / 8)
  referenceImages: string[];   // paths to generated frame images (start, mid?, end)
}

export interface SceneClip {
  sceneId: string;
  clipIndex: number;
  videoPath: string;
  durationSeconds: number; // always 8
}

export interface StitchedScene {
  sceneId: string;
  videoPath: string;
  trimmedDurationSeconds: number; // equals scene.audioDurationSeconds
}
```

---

## Scene Planner Service

```typescript
// scene/scene-planner.service.ts

@Injectable()
export class ScenePlannerService {
  private readonly CLIP_DURATION = 8; // Veo fixed clip duration in seconds

  planScenes(content: string, audioDurations: number[]): Scene[] {
    // audioDurations[i] = measured TTS duration for scene i in seconds
    // or estimate: ~2.5 words per second if TTS not yet generated

    return scenes.map((scene, i) => ({
      ...scene,
      audioDurationSeconds: audioDurations[i],
      clipsNeeded: Math.ceil(audioDurations[i] / this.CLIP_DURATION),
    }));
  }

  estimateDurationFromScript(script: string): number {
    const wordsPerSecond = 2.5;
    const wordCount = script.trim().split(/\s+/).length;
    return Math.ceil(wordCount / wordsPerSecond);
  }
}
```

---

## Frame Generator Service

```typescript
// image/frame-generator.service.ts

@Injectable()
export class FrameGeneratorService {

  async generateFramesForScene(scene: Scene): Promise<string[]> {
    const frames: string[] = [];

    if (scene.clipsNeeded === 1) {
      // Single clip: only need one reference image
      frames.push(await this.generateImage(scene.description, 'establishing'));
    } else {
      // Multi-clip: generate start, optional mid, end reference frames
      frames.push(await this.generateImage(scene.description, 'start'));

      // Generate mid frames for long scenes (>= 3 clips)
      for (let i = 1; i < scene.clipsNeeded - 1; i++) {
        frames.push(await this.generateImage(scene.description, `mid-${i}`));
      }

      frames.push(await this.generateImage(scene.description, 'end'));
    }

    return frames; // length === scene.clipsNeeded
  }

  private async generateImage(description: string, frameLabel: string): Promise<string> {
    // Call your image generation API (Imagen, DALL-E, etc.)
    // Return local file path to saved image
  }
}
```

---

## Veo Service

```typescript
// veo/veo.service.ts

@Injectable()
export class VeoService {

  async generateClip(options: {
    sceneDescription: string;
    referenceImagePath: string;
    previousClipLastFramePath?: string; // for chaining continuity
  }): Promise<string> {
    // Call Veo API with:
    // - image_start: previousClipLastFramePath ?? referenceImagePath
    // - prompt: sceneDescription
    // Returns path to downloaded 8s .mp4 clip

    // IMPORTANT: After generation, extract the last frame of this clip
    // so it can be passed as previousClipLastFramePath for the next clip
    await this.extractLastFrame(outputClipPath);

    return outputClipPath;
  }

  async extractLastFrame(clipPath: string): Promise<string> {
    const lastFramePath = clipPath.replace('.mp4', '_lastframe.jpg');
    // ffmpeg -sseof -0.1 -i clip.mp4 -frames:v 1 lastframe.jpg
    await execAsync(
      `ffmpeg -sseof -0.1 -i ${clipPath} -frames:v 1 ${lastFramePath} -y`
    );
    return lastFramePath;
  }
}
```

---

## Stitch Service

```typescript
// stitch/stitch.service.ts

@Injectable()
export class StitchService {

  async stitchAndTrim(
    clipPaths: string[],
    targetDurationSeconds: number,
    outputPath: string,
  ): Promise<string> {
    if (clipPaths.length === 1) {
      // Only trim, no concat needed
      return this.trimClip(clipPaths[0], targetDurationSeconds, outputPath);
    }

    const concatPath = outputPath.replace('.mp4', '_concat.mp4');
    await this.concatClips(clipPaths, concatPath);
    return this.trimClip(concatPath, targetDurationSeconds, outputPath);
  }

  private async concatClips(clipPaths: string[], outputPath: string): Promise<void> {
    // Write ffmpeg concat list file
    const listContent = clipPaths.map(p => `file '${p}'`).join('\n');
    const listPath = outputPath.replace('.mp4', '_list.txt');
    await fs.writeFile(listPath, listContent);

    // ffmpeg concat (re-encode for clean joins)
    await execAsync(
      `ffmpeg -f concat -safe 0 -i ${listPath} -c:v libx264 -preset fast ${outputPath} -y`
    );
  }

  private async trimClip(
    inputPath: string,
    durationSeconds: number,
    outputPath: string,
  ): Promise<string> {
    await execAsync(
      `ffmpeg -i ${inputPath} -t ${durationSeconds} -c copy ${outputPath} -y`
    );
    return outputPath;
  }
}
```

---

## Main Orchestrator (VideoService)

```typescript
// video.service.ts

@Injectable()
export class VideoService {
  constructor(
    private scenePlanner: ScenePlannerService,
    private frameGenerator: FrameGeneratorService,
    private veoService: VeoService,
    private stitchService: StitchService,
  ) {}

  async generateVideo(content: string): Promise<string> {
    // Step 1: Plan scenes with durations
    const rawScenes = await this.extractScenes(content);
    const audioDurations = rawScenes.map(s =>
      this.scenePlanner.estimateDurationFromScript(s.script)
    );
    const scenes = this.scenePlanner.planScenes(content, audioDurations);

    const stitchedScenes: StitchedScene[] = [];

    for (const scene of scenes) {
      // Step 2: Generate reference images for this scene
      scene.referenceImages = await this.frameGenerator.generateFramesForScene(scene);

      // Step 3: Generate N clips for this scene with frame chaining
      const clipPaths: string[] = [];
      let previousLastFramePath: string | undefined;

      for (let i = 0; i < scene.clipsNeeded; i++) {
        const clipPath = await this.veoService.generateClip({
          sceneDescription: scene.description,
          referenceImagePath: scene.referenceImages[i] ?? scene.referenceImages[0],
          previousClipLastFramePath: previousLastFramePath,
        });

        clipPaths.push(clipPath);

        // Chain: extract last frame for next clip's start reference
        previousLastFramePath = await this.veoService.extractLastFrame(clipPath);
      }

      // Step 4: Stitch clips and trim to exact audio duration
      const stitchedPath = await this.stitchService.stitchAndTrim(
        clipPaths,
        scene.audioDurationSeconds,
        `output/scene_${scene.sceneIndex}.mp4`,
      );

      stitchedScenes.push({
        sceneId: scene.id,
        videoPath: stitchedPath,
        trimmedDurationSeconds: scene.audioDurationSeconds,
      });
    }

    // Step 5: Concatenate all scene videos into final output
    const finalPath = await this.stitchService.stitchAndTrim(
      stitchedScenes.map(s => s.videoPath),
      stitchedScenes.reduce((sum, s) => sum + s.trimmedDurationSeconds, 0),
      'output/final.mp4',
    );

    return finalPath;
  }
}
```

---

## Summary of Changes vs Old Approach

| Aspect | Old | New |
|---|---|---|
| Clips per scene | Always 1 | `ceil(audioDuration / 8)` |
| Images per scene | 1 reference image | N images (start, mid, end) |
| Clip chaining | None | Last frame → next clip's start |
| Duration tracking | None | Per-scene audio duration |
| Stitch step | None | ffmpeg concat + trim per scene |
| Audio sync | Broken mid-scene | Trimmed to exact audio length |

---

## Notes for Cursor

- All services are injectable and can be swapped or mocked independently
- The `VeoService.generateClip` options should be extended to match your actual Veo API client interface
- If you have real TTS audio files, replace `estimateDurationFromScript` with actual audio duration measurement using `ffprobe`
- `extractLastFrame` requires ffmpeg to be installed and accessible in the runtime environment
- Consider adding a queue (Bull/BullMQ) around `VeoService.generateClip` calls since Veo generation is slow and rate-limited
