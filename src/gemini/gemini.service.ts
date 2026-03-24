import { HttpException, Injectable } from "@nestjs/common";
import { ConfigService } from "@nestjs/config";
import { GoogleGenAI } from "@google/genai";
import * as path from "path";
import * as fs from "fs/promises";
import { spawn } from "child_process";

function sleep(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Veo 3.x text-to-video (Gemini API): supported clip lengths are 4, 6, or 8 seconds — always use 8.
 * @see https://ai.google.dev/gemini-api/docs/video
 */
const VEO_MAX_CREATE_DURATION_SECONDS = 8;

/**
 * Each Veo extension call appends this many seconds (fixed increment for the extension API).
 */
const VEO_EXTENSION_DURATION_SECONDS = 7;

@Injectable()
export class GeminiService {
    private readonly ai: GoogleGenAI;

    constructor(private readonly config: ConfigService) {
        const apiKey = this.config.get<string>("GEMINI_API_KEY") || this.config.get<string>("GOOGLE_API_KEY") || process.env.GEMINI_API_KEY;
        // If you prefer env auto-pickup, you can do new GoogleGenAI({})
        // but being explicit is clearer for NestJS server apps.
        this.ai = new GoogleGenAI(apiKey ? { apiKey } : {});
    }


    async geminiVideoScript(content: string) {
        const systemInstruction = `
    You are an expert cinematic educational scene designer.

        Your task is to convert LMS instructional content into structured scene-based video prompts.

        Output Requirements:

        1. Return ONLY a valid JSON array of strings.
        2. Do NOT wrap in markdown code fences.
        3. Do NOT return an object. Do NOT include keys.
        4. Each string must describe ONE visual scene.
        5. Each scene must be cinematic, visually descriptive, realistic, and suitable for AI video generation.
        6. Include environment details (lab, classroom, outdoor, etc.).
        7. Mention lighting (natural light, soft lighting, warm classroom light, etc.).
        8. Add camera movement (slow zoom, macro shot, close-up, wide shot, tracking shot, etc.).
        9. Maintain an educational tone.
        10. Optionally include subtle background narration style such as:
        - calm instructional voice-over
        - clear educational explanation tone
        - steady classroom narration
        11. Do NOT include spoken dialogue between characters.
        12. Keep scenes visually rich but concise.
        13. Ensure the flow logically follows the instructional steps.
        14. Create 5–8 scenes depending on complexity.

        Output Example:

        [
        "Scene description with cinematic visuals and calm educational voice-over tone...",
        "Next scene..."
        ]
      `;
      
        const result: any = await this.ai.models.generateContent({
          model: "gemini-2.5-flash",
          contents: [{ role: "user", parts: [{ text: content }] }],
          config: {
            temperature: 0.8,
            topP: 0.9,
            maxOutputTokens: 2048,
            systemInstruction,
            // If your SDK supports it, keep response as JSON:
            // responseMimeType: "application/json"
          },
        });
      
        const rawText =
          result?.candidates?.[0]?.content?.parts?.map((p: any) => p.text || "").join("") || "";
      
        // 1) Strip ```json fences if model still sends them
        const cleaned = rawText
          .replace(/```json\s*/gi, "")
          .replace(/```\s*/g, "")
          .trim();
      
        // 2) Parse the JSON array
        const parsed = JSON.parse(cleaned);
      
        // 3) Ensure it's an array of strings
        if (!Array.isArray(parsed) || !parsed.every((x) => typeof x === "string")) {
          throw new Error("Gemini did not return a JSON array of strings.");
        }
      
        return parsed; // <-- returns string[]
      }

    /**
     * Maps the same scene list style as `/scene` (`geminiVideoScript`) onto the exact segment
     * timeline used by Veo create + extension (one prompt per segment duration).
     */
    private async veoSegmentPromptsFromScenes(
        content: string,
        scenes: string[],
        segmentSeconds: number[],
    ): Promise<string[]> {
        const segmentCount = segmentSeconds.length;
        const durationsText = segmentSeconds.map((s, i) => `Segment ${i + 1}: ${s}s`).join("\n");

        const systemInstruction = `
You are an expert instructional video segment designer for Veo video extension.

You are given:
- The original LMS/instructional content.
- A JSON array of scene descriptions that were produced from that content (same pipeline as a montage: cinematic, educational, lighting/camera rich, no character dialogue).

Your task:
1) Return ONLY a valid JSON array of strings. No markdown, no extra keys.
2) The array MUST contain exactly ${segmentCount} strings (no more, no less).
3) Entry index i (0-based) is the prompt for segment (i+1) with the i-th duration in the list above — cover the full timeline in order.
4) Reuse and expand the scene material across all segments: distribute the instructional narrative across the timeline; maintain continuity (same characters/style/lighting/camera language as the input scenes).
5) Each segment prompt must describe what is visible and how the shot evolves during that segment only; include camera movement where appropriate.
6) Keep an educational tone. Do NOT write spoken dialogue between characters.
7) Segments must connect as one continuous extended video (Veo will extend from the previous segment).

Segment durations (in order):
${durationsText}
`;

        const userPayload = `Original instructional content:\n\n${content}\n\n---\nScene list from script step (JSON array of strings):\n${JSON.stringify(scenes)}\n\nProduce exactly ${segmentCount} segment prompts for Veo as specified.`;

        const result: any = await this.ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: [{ role: "user", parts: [{ text: userPayload }] }],
            config: {
                temperature: 0.8,
                topP: 0.9,
                maxOutputTokens: 8192,
                systemInstruction,
            },
        });

        const rawText =
            result?.candidates?.[0]?.content?.parts?.map((p: any) => p.text || "").join("") || "";

        const cleaned = rawText
            .replace(/```json\s*/gi, "")
            .replace(/```\s*/g, "")
            .trim();

        const parsed = JSON.parse(cleaned);
        if (!Array.isArray(parsed) || !parsed.every((x) => typeof x === "string")) {
            throw new Error("Veo segment expansion did not return a JSON array of strings.");
        }
        if (parsed.length !== segmentCount) {
            throw new Error(`Veo segment expansion length mismatch. Expected ${segmentCount}, got ${parsed.length}.`);
        }
        return parsed;
    }

    private wrapSegmentPrompt(args: { segmentIndex: number; segmentCount: number; segmentSeconds: number; prompt: string }) {
        const { segmentIndex, segmentCount, segmentSeconds, prompt } = args;
        return `Segment ${segmentIndex} of ${segmentCount} (${segmentSeconds}s). Keep same character/style/lighting continuity. ${prompt.trim()}`;
    }

    private async pollVeoOperation(operation: any, maxAttempts = 180, pollMs = 10_000) {
        let attempt = 0;
        while (!operation.done) {
            attempt += 1;
            if (attempt > maxAttempts) {
                throw new Error("Veo generation timed out while polling operation status.");
            }
            await sleep(pollMs);
            operation = await this.ai.operations.getVideosOperation({ operation });
        }
        return operation;
    }

    /** Keep only stable file identity fields used by Veo extension. */
    private toVeoVideoRef(video: any): { name?: string; uri?: string } {
        if (!video || typeof video !== "object") return {};
        const ref: { name?: string; uri?: string } = {}
        if (typeof video.name === "string" && video.name.trim().length > 0) {
            ref.name = video.name;
        }
        if (typeof video.uri === "string" && video.uri.trim().length > 0) {
            ref.uri = video.uri;
        }
        return ref; // returns both if both exist
    }

    private isVeoExtensionEligibilityError(err: unknown): boolean {
        const message =
            err instanceof Error
                ? err.message
                : typeof err === "string"
                    ? err
                    : JSON.stringify(err ?? "");
        // API wording varies: "for extension" vs "that has been processed" — both mean bad input ref / not ready.
        return (
            /video must be generated by veo for extension/i.test(message) ||
            /has been processed/i.test(message) ||
            /must be a video that was generated by veo/i.test(message)
        );
    }

    /** Extension calls should use stable file `name` only; mixing `uri` can confuse validation. */
    private videoRefForExtension(ref: { name?: string; uri?: string }): { name: string } | { uri: string } {
        if (ref.name?.trim()) return { name: ref.name.trim() };
        if (ref.uri?.trim()) return { uri: ref.uri.trim() };
        return { name: "" };
    }

    private async extendVeoSegmentWithRetry(args: {
        model: string;
        video: any;
        prompt: string;
        aspectRatio: "9:16" | "16:9";
        extensionSeconds: number;
        segmentLabel: string;
    }) {
        const { model, video, prompt, aspectRatio, segmentLabel } = args;
        let videoRef = await this.waitForProcessedVeoVideo(video, `${segmentLabel} source`);

        const runExtend = async (ref: { name?: string; uri?: string }) => {
            const videoInput = this.videoRefForExtension(ref);
            if ("name" in videoInput && !videoInput.name) {
                throw new Error(`[Veo][2min] ${segmentLabel}: missing file name for extension input.`);
            }
            // SDK `Video` typing is uri/bytes-only; Gemini file refs use `name` at runtime.
            return this.ai.models.generateVideos({
                model,
                video: videoInput as any,
                prompt,
                // For extension calls, Veo derives segment behavior from the source video + prompt.
                // Avoid sending durationSeconds here to prevent API argument validation regressions.
                config: {
                    numberOfVideos: 1,
                    resolution: "720p",
                    aspectRatio,
                },
            });
        };

        const maxEligibilityRetries = 8;
        let lastErr: unknown;
        for (let attempt = 1; attempt <= maxEligibilityRetries; attempt++) {
            try {
                return await runExtend(videoRef);
            } catch (err) {
                lastErr = err;
                if (!this.isVeoExtensionEligibilityError(err)) {
                    throw err;
                }
                const name = videoRef.name?.trim();
                if (!name) {
                    throw err;
                }
                const backoffMs = Math.min(60_000, 5_000 * attempt);
                console.warn(
                    `[Veo][2min] ${segmentLabel}: extension not accepted (processed/eligibility). attempt=${attempt}/${maxEligibilityRetries} sleeping ${backoffMs}ms then re-fetch file.`,
                );
                await sleep(backoffMs);
                videoRef = await this.waitForProcessedVeoVideo({ name }, `${segmentLabel} eligibility-retry-${attempt}`);
            }
        }
        throw lastErr;
    }

    /**
     * Veo can report operation done before the produced file becomes extend-eligible.
     * Poll file state by name for a short window to reduce "must be processed" errors.
     */
    private async waitForProcessedVeoVideo(
        video: any,
        contextLabel: string,
        maxAttempts = 36,
        pollMs = 5_000,
        stabilizationAfterActiveMs = 3_000,
    ): Promise<{ name?: string; uri?: string }> {
        let ref = this.toVeoVideoRef(video);
        if (!ref.name) return ref;

        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            const fileObj = await (this.ai as any).files.get({ name: ref.name });
            const rawState = String(fileObj?.state ?? "");
            const state = rawState.toUpperCase();
            ref = this.toVeoVideoRef(fileObj);

            // GenAI FileState: PROCESSING | ACTIVE | FAILED | STATE_UNSPECIFIED
            if (state === "ACTIVE") {
                // Backend can still reject extend briefly after ACTIVE; small settle window helps.
                if (stabilizationAfterActiveMs > 0) {
                    console.log(
                        `[Veo][2min] ${contextLabel}: file ACTIVE; stabilization wait ${stabilizationAfterActiveMs}ms before use.`,
                    );
                    await sleep(stabilizationAfterActiveMs);
                }
                return { name: ref.name };
            }

            if (state === "FAILED") {
                throw new Error(`[Veo][2min] ${contextLabel}: file entered terminal state ${state}.`);
            }

            // PROCESSING, STATE_UNSPECIFIED, empty: keep polling (do not treat as extend-ready).
            if (attempt < maxAttempts) {
                console.log(
                    `[Veo][2min] ${contextLabel}: waiting for ACTIVE (extend-ready) state=${rawState || "(empty)"} attempt=${attempt}/${maxAttempts}`,
                );
                await sleep(pollMs);
            }
        }

        throw new Error(
            `[Veo][2min] ${contextLabel}: timed out after ${maxAttempts} attempts waiting for file state ACTIVE (still not extend-ready).`,
        );
    }

    /** Best-effort extraction of provider error payload for API responses/logging. */
    private extractStructuredError(err: unknown): any {
        if (err && typeof err === "object") {
            const anyErr = err as any;
            if (anyErr.error && typeof anyErr.error === "object") return anyErr;
            if (anyErr.response && typeof anyErr.response === "object") return anyErr.response;
        }

        if (err instanceof Error) {
            const msg = err.message?.trim();
            if (msg?.startsWith("{") || msg?.startsWith("[")) {
                try {
                    return JSON.parse(msg);
                } catch {
                    return { message: msg };
                }
            }
            return { message: msg };
        }

        if (typeof err === "string") {
            const msg = err.trim();
            if (msg.startsWith("{") || msg.startsWith("[")) {
                try {
                    return JSON.parse(msg);
                } catch {
                    return { message: msg };
                }
            }
            return { message: msg };
        }

        try {
            return JSON.parse(JSON.stringify(err));
        } catch {
            return { message: String(err) };
        }
    }

    /**
     * Persist latest successful Veo output when an extension chain fails mid-way.
     * This protects progress so long-running jobs do not lose all work.
     */
    private async persistFailureCheckpoint(args: {
        outDir: string;
        currentVideo: { name?: string; uri?: string } | null;
        segmentCompleted: number;
        segmentCount: number;
    }): Promise<{ filePath: string; fileName: string } | null> {
        const { outDir, currentVideo, segmentCompleted, segmentCount } = args;
        if (!currentVideo?.name && !currentVideo?.uri) return null;

        const checkpointAtMs = Date.now();
        const fileName = `checkpoint_segment_${segmentCompleted}_of_${segmentCount}_${checkpointAtMs}.mp4`;
        const filePath = path.join(outDir, fileName);

        await this.ai.files.download({
            file: currentVideo,
            downloadPath: filePath,
        });

        console.log(`[Veo][2min] checkpoint saved at failure -> ${filePath}`);
        return { filePath, fileName };
    }

    /** Shared Veo config: max resolution we use here + explicit duration (create or extend). */
    private veoGenerateVideosConfig(
        aspectRatio: "9:16" | "16:9",
        durationSeconds: number,
        includeDurationSeconds = true,
    ): { aspectRatio: "9:16" | "16:9"; resolution: "720p"; numberOfVideos: number; durationSeconds?: number } {
        const config: { aspectRatio: "9:16" | "16:9"; resolution: "720p"; numberOfVideos: number; durationSeconds?: number } = {
            aspectRatio,
            resolution: "720p",
            numberOfVideos: 1,
        };
        if (includeDurationSeconds) {
            config.durationSeconds = durationSeconds;
        }
        return config;
    }

    async generateVideoToFile(prompt: string, aspectRatio: "9:16" | "16:9") {
        // ensure output dir exists
        const outDir = path.resolve(process.cwd(), "generated");
        await fs.mkdir(outDir, { recursive: true });

        const fileName = `veo_${Date.now()}.mp4`;
        const filePath = path.join(outDir, fileName);

        console.log("filepath", filePath)

        try {
            // 1) start generation
            let operation = await this.ai.models.generateVideos({
                model: "veo-3.1-generate-preview",
                prompt,
                config: this.veoGenerateVideosConfig(aspectRatio, VEO_MAX_CREATE_DURATION_SECONDS),
            });
            console.log("operation", operation);

            // 2) poll until done (add a max attempts safety)
            const maxAttempts = 120; // 120 * 10s = 20 mins
            let attempt = 0;

            while (!operation.done) {
                attempt += 1;
                if (attempt > maxAttempts) {
                    throw new Error("Veo generation timed out while polling operation status.");
                }
                await sleep(10_000);
                operation = await this.ai.operations.getVideosOperation({ operation });
            }

            // 3) validate response + download
            const generated = operation.response?.generatedVideos?.[0];
            if (!generated?.video) {
                throw new Error("No generated video found in operation response.");
            }

            // This matches Google’s JS example: files.download({ file, downloadPath })
            // (downloadPath can be relative or absolute)
            let download = await this.ai.files.download({
                file: generated.video,
                downloadPath: filePath,
            });
            console.log("download", download)
            let to_return = { filePath, fileName }
            return to_return;
        } catch (error) {
            console.log("error while creating video", error.message);
        }

    }

    /**
     * One API that generates ~2 minutes by chaining Veo extensions (no local stitching).
     *
     * Max-duration strategy (Gemini Veo limits):
     * - First segment: {@link VEO_MAX_CREATE_DURATION_SECONDS}s (max allowed for text-to-video: 4/6/8).
     * - Each extension: {@link VEO_EXTENSION_DURATION_SECONDS}s per call (extension increment).
     * - ~120s total: 8 + 16×7 = 120.
     */
    async generateScenesAndExtendToTwoMinutes(content: string, aspectRatio: "9:16" | "16:9") {
        const outDir = path.resolve(process.cwd(), "generated");
        await fs.mkdir(outDir, { recursive: true });

        const targetSeconds = 120;
        const initialSeconds = VEO_MAX_CREATE_DURATION_SECONDS;
        const extensionSeconds = VEO_EXTENSION_DURATION_SECONDS;
        const extensionCount = Math.floor((targetSeconds - initialSeconds) / extensionSeconds); // 16

        const segmentSeconds = [initialSeconds, ...Array(extensionCount).fill(extensionSeconds)];
        const segmentCount = segmentSeconds.length; // 17

        console.log(`[Veo][2min] segmentSeconds=${JSON.stringify(segmentSeconds)}`);

        // Same script step as POST /gemini/scene: cinematic scene array from content, then map to timed segments.
        const scenes = await this.geminiVideoScript(content);
        console.log(`[Veo][2min] geminiVideoScript scenes count=${scenes.length}`);

        const segmentPrompts = await this.veoSegmentPromptsFromScenes(content, scenes, segmentSeconds);
        console.log(`[Veo][2min] generated ${segmentPrompts.length} segment prompts from scenes`);

        let currentVideo: { name?: string; uri?: string } | null = null;
        let completedSegments = 0;

        try {
            // 1) Create first segment video with maximum create duration.
            const firstPrompt = this.wrapSegmentPrompt({
                segmentIndex: 1,
                segmentCount,
                segmentSeconds: segmentSeconds[0],
                prompt: segmentPrompts[0],
            });

            console.log(`[Veo][2min] creating segment 1 (${segmentSeconds[0]}s)...`);
            let operation = await this.ai.models.generateVideos({
                model: "veo-3.1-generate-preview",
                prompt: firstPrompt,
                config: this.veoGenerateVideosConfig(aspectRatio, initialSeconds),
            });

            operation = await this.pollVeoOperation(operation);
            const firstGenerated = operation.response?.generatedVideos?.[0];
            if (!firstGenerated?.video) {
                throw new Error("No generated video found in first segment operation response.");
            }

            currentVideo = await this.waitForProcessedVeoVideo(firstGenerated.video, "segment 1 output");
            completedSegments = 1;

            // 2) Extend for remaining segments
            for (let i = 1; i < segmentCount; i++) {
                const prompt = this.wrapSegmentPrompt({
                    segmentIndex: i + 1,
                    segmentCount,
                    segmentSeconds: segmentSeconds[i],
                    prompt: segmentPrompts[i],
                });

                console.log(`[Veo][2min] extending segment ${i + 1}/${segmentCount} for ${extensionSeconds}s...`);
                operation = await this.extendVeoSegmentWithRetry({
                    model: "veo-3.1-generate-preview",
                    video: currentVideo,
                    prompt,
                    aspectRatio,
                    extensionSeconds,
                    segmentLabel: `segment ${i + 1}/${segmentCount}`,
                });

                operation = await this.pollVeoOperation(operation);

                const generated = operation.response?.generatedVideos?.[0];
                if (!generated?.video) {
                    throw new Error(`No generated video found in extension operation response for segment ${i + 1}.`);
                }

                currentVideo = await this.waitForProcessedVeoVideo(generated.video, `segment ${i + 1} output`);
                completedSegments = i + 1;
            }

            // 3) Download only the final extended output; filename = task completion time (ms epoch).
            const completedAtMs = Date.now();
            const fileName = `${completedAtMs}.mp4`;
            const filePath = path.join(outDir, fileName);

            console.log(`[Veo][2min] downloading final extended video -> ${filePath}`);
            await this.ai.files.download({
                file: currentVideo,
                downloadPath: filePath,
            });

            console.log(`[Veo][2min] downloaded final extended video -> ${filePath}`);

            return {
                filePath,
                fileName,
                completedAtMs,
                segmentSeconds,
                /** Same intermediate array as POST /gemini/scene (`geminiVideoScript`). */
                scenes,
            };
        } catch (error: any) {
            let checkpoint: { filePath: string; fileName: string } | null = null;
            try {
                checkpoint = await this.persistFailureCheckpoint({
                    outDir,
                    currentVideo,
                    segmentCompleted: completedSegments,
                    segmentCount,
                });
            } catch (checkpointErr: any) {
                console.error("[Veo][2min] failed to persist checkpoint at error time:", checkpointErr?.message || checkpointErr);
            }

            const originalMessage =
                error instanceof Error ? error.message : typeof error === "string" ? error : JSON.stringify(error);
            const structuredError = this.extractStructuredError(error);

            throw new HttpException(
                {
                    statusCode: 502,
                    error: "Bad Gateway",
                    message: `Veo extension failed after segment ${completedSegments}/${segmentCount}.`,
                    stage: "generateScenesAndExtendToTwoMinutes",
                    segmentCompleted: completedSegments,
                    segmentCount,
                    checkpoint,
                    originalErrorMessage: originalMessage,
                    upstreamError: structuredError,
                },
                502,
            );
        }
    }


    /**
     * Generates a single clip and saves it to `downloadPath`.
     */
    async generateSingleClipToFile(
        scenes: string[],
        aspectRatio: string,
        outputPaths?: string[], // ✅ NEW
    ) {
        console.log("entered generateSingleClipToFile");

        try {
            const collection: string[] = [];
            console.log("prompts", scenes);

            for (let i = 0; i < scenes.length; i++) {
                const prompt = scenes[i].trim();
                console.log("prompt from scene", prompt);

                let operation = await this.ai.models.generateVideos({
                    model: "veo-3.1-generate-preview",
                    prompt,
                    config: this.veoGenerateVideosConfig(
                        aspectRatio as "9:16" | "16:9",
                        VEO_MAX_CREATE_DURATION_SECONDS,
                    ),
                });
                const maxAttempts = 120;
                let attempt = 0;

                while (!operation.done) {
                    attempt += 1;
                    if (attempt > maxAttempts) {
                        throw new Error("Veo generation timed out while polling operation status.");
                    }
                    await sleep(10_000);
                    operation = await this.ai.operations.getVideosOperation({ operation });
                }
                console.log("operation", operation);

                const generated = operation.response?.generatedVideos?.[0];
                console.log("generated", generated);
                if (!generated?.video) {
                    throw new Error("No generated video found in operation response.");
                }

                const outDir = path.resolve(process.cwd(), "generated");
                await fs.mkdir(outDir, { recursive: true });

                // ✅ If montage passes an output path, use it. Otherwise fallback to timestamp name.
                const downloadPath =
                    outputPaths?.[i] ??
                    path.join(outDir, `veo_${Date.now()}_${i + 1}.mp4`);

                await this.ai.files.download({
                    file: generated.video,
                    downloadPath,
                });

                console.log("downloadPath", downloadPath);
                collection.push(downloadPath);
            }

            return collection;
        } catch (error: any) {
            console.log("error while creating video", error.message, error);
            throw error; // ✅ important so controller gets proper error response
        }
    }

    /**
     * Concats MP4 clips into one MP4 using FFmpeg concat demuxer.
     * This requires that all clips have compatible encoding params.
     * (Usually true if all generated by Veo with same aspect ratio.)
     */
    private async concatClipsFFmpeg(clipPaths: string[], outputPath: string) {
        // Create a concat list file
        // Important: use absolute paths and escape single quotes
        const listFilePath = path.join(path.dirname(outputPath), `concat_${Date.now()}.txt`);
        const fileLines = clipPaths
            .map((p) => `file '${p.replace(/'/g, "'\\''")}'`)
            .join("\n");

        await fs.writeFile(listFilePath, fileLines, "utf8");

        // ffmpeg -f concat -safe 0 -i list.txt -c copy output.mp4
        await new Promise<void>((resolve, reject) => {
            const args = [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                listFilePath,
                "-c",
                "copy",
                outputPath,
            ];

            const ff = spawn("ffmpeg", args, { stdio: "pipe" });

            let stderr = "";
            ff.stderr.on("data", (d) => (stderr += d.toString()));

            ff.on("error", (err) => reject(err));

            ff.on("close", async (code) => {
                // cleanup list file
                await fs.unlink(listFilePath).catch(() => { });
                if (code === 0) return resolve();

                // If stream copy fails because clips differ slightly,
                // we can fall back to re-encode concat (slower but robust).
                reject(new Error(`FFmpeg concat failed (code=${code}). Details:\n${stderr}`));
            });
        });
    }

    /**
     * Fallback concat that re-encodes (more compatible, slower).
     * Uses filter_complex concat.
     */
    private async concatClipsFFmpegReencode(clipPaths: string[], outputPath: string) {
        await new Promise<void>((resolve, reject) => {
            // Build: -i clip1 -i clip2 ... then filter_complex concat
            const inputs: string[] = [];
            clipPaths.forEach((p) => {
                inputs.push("-i", p);
            });

            // concat filter:
            // [0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[outv][outa]
            const pairs = clipPaths
                .map((_, i) => `[${i}:v:0][${i}:a:0]`)
                .join("");
            const filter = `${pairs}concat=n=${clipPaths.length}:v=1:a=1[outv][outa]`;

            const args = [
                ...inputs,
                "-filter_complex",
                filter,
                "-map",
                "[outv]",
                "-map",
                "[outa]",
                // output encoding - tune as you wish
                "-movflags",
                "+faststart",
                outputPath,
            ];

            const ff = spawn("ffmpeg", args, { stdio: "pipe" });

            let stderr = "";
            ff.stderr.on("data", (d) => (stderr += d.toString()));

            ff.on("error", (err) => reject(err));
            ff.on("close", (code) => {
                if (code === 0) return resolve();
                reject(new Error(`FFmpeg re-encode concat failed (code=${code}). Details:\n${stderr}`));
            });
        });
    }

    /**
     * Public API: scenes[] => generate clip1..clipN sequentially => concat => return combined file.
     */
    async generateMontageToFile(scenes: string[], aspectRatio: string) {
        const outDir = path.resolve(process.cwd(), "generated");
        await fs.mkdir(outDir, { recursive: true });

        const montageId = `montage_${Date.now()}`;
        const montageDir = path.join(outDir, montageId);
        await fs.mkdir(montageDir, { recursive: true });

        // ✅ create exact output paths upfront
        const clipFiles = scenes.map((_, i) =>
            path.join(montageDir, `clip_${String(i + 1).padStart(2, "0")}.mp4`)
        );

        // ✅ generate ALL clips into those exact file names
        const prompts = scenes.map(
            (scenePrompt, i) =>
                `Scene ${i + 1} of ${scenes.length}. Keep same character/style/lighting. ${scenePrompt}`
        );

        await this.generateSingleClipToFile(prompts, aspectRatio, clipFiles);

        // ✅ concat result
        const combinedFileName = `${montageId}.mp4`;
        const combinedFilePath = path.join(outDir, combinedFileName);

        try {
            await this.concatClipsFFmpeg(clipFiles, combinedFilePath);
        } catch (err) {
            await this.concatClipsFFmpegReencode(clipFiles, combinedFilePath);
        }

        console.log("combinedFilePath", combinedFilePath);
        return {
            combinedFilePath,
            combinedFileName,
            clipFiles,
        };
    }
}