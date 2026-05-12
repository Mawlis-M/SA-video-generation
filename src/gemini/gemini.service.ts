import { Injectable } from "@nestjs/common";
import { ConfigService } from "@nestjs/config";
import { GoogleGenAI, Modality } from "@google/genai";
import OpenAI from "openai";
import * as path from "path";
import * as fs from "fs/promises";
import { spawn } from "child_process";

function sleep(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

/** OpenAI Responses API: prefer `output_text`, fall back to message content parts (same pattern as document-topics). */
function extractOpenAIResponseOutputText(response: unknown): string {
    const r = response as {
        output_text?: string;
        output?: { content?: { text?: string; type?: string }[] }[];
    };
    const top = r?.output_text?.trim();
    if (top) return top;
    const nested =
        r?.output
            ?.flatMap((item) => item.content ?? [])
            .filter((part) => part.type === "output_text" && !!part.text)
            .map((part) => part.text ?? "")
            .join("") ?? "";
    return nested.trim();
}

const FLASH_IMAGE_MODEL = "gemini-2.5-flash-image";
const IMAGEN_GENERATE_MODEL = "imagen-3.0-generate-002";
const VEO_MODEL = "veo-3.1-generate-preview";
const OPENAI_SCENE_MODEL = "gpt-5";

/** Veo sometimes returns INTERNAL (gRPC 13) on a finished LRO; short backoff + full clip retry usually succeeds. */
const VEO_MONTAGE_CLIP_MAX_ATTEMPTS = 5;
const VEO_MONTAGE_CLIP_RETRY_BASE_MS = 25_000;
const PIPELINE_STEP_MAX_RETRIES = 5;
const PIPELINE_STEP_RETRY_BASE_MS = 2_000;

/** Duration-aware document-to-video pipeline (see APPROACH.md). */
const VEO_CLIP_DURATION_SECONDS = 8;
const NARRATION_WORDS_PER_SECOND = 2.5;

/**
 * One planned scene in the duration-aware pipeline. Mirrors APPROACH.md `Scene` interface.
 * `clipsNeeded` is derived from the narration script length so a long scene no longer
 * gets clipped at 8 seconds — it spans `clipsNeeded` chained Veo clips.
 */
export interface PlannedScene {
    id: string;
    sceneIndex: number; // 0-based
    description: string;
    script: string;
    audioDurationSeconds: number;
    clipsNeeded: number;
    referenceImages: string[]; // populated as keyframes get generated
}

export interface StitchedScene {
    sceneId: string;
    sceneIndex: number;
    videoPath: string;
    /** Stitched scene duration in seconds (ffprobe when available; else narration estimate). */
    trimmedDurationSeconds: number;
}

export type SceneImageMontageFailureDetails = {
    step: string;
    montageId: string;
    montageDir: string;
    imagesDir: string;
    retriesPerStep: number;
    successfulImageCount: number;
    successfulClipCount: number;
    partialCombinedFilePath: string | null;
    partialCombinedFileName: string | null;
    originalErrorMessage: string;
};

export type SceneImageMontageResumeOptions = {
    /** Reuse an existing montage workspace folder id, e.g. `montage_img_1777...`. */
    resumeMontageId?: string;
    /** 1-based scene index to resume from. Example: 26 means continue from scene_26 / clip_26. */
    resumeFromScene?: number;
};

export class SceneImageMontageError extends Error {
    readonly details: SceneImageMontageFailureDetails;

    constructor(message: string, details: SceneImageMontageFailureDetails) {
        super(message);
        this.name = "SceneImageMontageError";
        this.details = details;
    }
}

@Injectable()
export class GeminiService {
    private readonly ai: GoogleGenAI;
    private readonly openai: OpenAI;
    private readonly sceneModel: string;
    private ffmpegChecked = false;
    private ffmpegAvailable = false;

    constructor(private readonly config: ConfigService) {
        const apiKey = this.config.get<string>("GEMINI_API_KEY") || this.config.get<string>("GOOGLE_API_KEY") || process.env.GEMINI_API_KEY;
        const openaiApiKey = this.config.get<string>("OPENAI_API_KEY") || process.env.OPENAI_API_KEY;
        // If you prefer env auto-pickup, you can do new GoogleGenAI({})
        // but being explicit is clearer for NestJS server apps.
        this.ai = new GoogleGenAI(apiKey ? { apiKey } : {});
        this.openai = new OpenAI({ apiKey: openaiApiKey });
        this.sceneModel =
            this.config.get<string>("OPENAI_SCENE_MODEL") ||
            process.env.OPENAI_SCENE_MODEL ||
            OPENAI_SCENE_MODEL;
    }

    /**
     * Fail fast when ffmpeg is not on PATH. The montage pipeline only concats at the very end,
     * so without this check we'd spend money on every Veo clip and then crash on stitching.
     */
    private async assertFfmpegAvailable(): Promise<void> {
        if (this.ffmpegChecked) {
            if (!this.ffmpegAvailable) {
                throw new Error(
                    "ffmpeg is not installed or not on PATH. Install ffmpeg and ensure `ffmpeg -version` works in the shell that starts this server before running the video pipeline.",
                );
            }
            return;
        }
        this.ffmpegChecked = true;
        await new Promise<void>((resolve, reject) => {
            const ff = spawn("ffmpeg", ["-version"], { stdio: "ignore" });
            ff.on("error", (err: NodeJS.ErrnoException) => {
                if (err.code === "ENOENT") {
                    reject(
                        new Error(
                            "ffmpeg is not installed or not on PATH. Install ffmpeg and ensure `ffmpeg -version` works in the shell that starts this server before running the video pipeline.",
                        ),
                    );
                } else {
                    reject(err);
                }
            });
            ff.on("close", (code) => {
                if (code === 0) {
                    this.ffmpegAvailable = true;
                    resolve();
                } else {
                    reject(new Error(`ffmpeg -version exited with code ${code}. Reinstall ffmpeg.`));
                }
            });
        });
    }


    private static readonly VIDEO_SCRIPT_MAX_RETRIES = 3;

    async geminiVideoScript(content: string) {
        const systemInstruction = `
    You are an expert cinematic educational scene designer.

Your task is to convert a detailed instructional summary into structured scene-based video prompts for educational or demonstration videos.

Output Requirements:

1. Return ONLY a valid JSON array of strings.
2. Do NOT wrap in markdown code fences.
3. Do NOT return an object. Do NOT include keys.
4. Each string must describe ONE visual scene covering a single instructional step, concept, or topic segment.
5. Each scene must be cinematic, visually descriptive, realistic, and suitable for AI video generation.
6. Include environment details relevant to the content (lab, classroom, workshop, office, outdoor field, factory floor, etc.).
7. Mention lighting conditions appropriate to the setting (natural daylight, soft diffused lab lighting, warm overhead classroom light, cool industrial lighting, etc.).
8. Add a camera movement or shot type for every scene (slow zoom in, macro close-up, wide establishing shot, tracking shot alongside subject, overhead bird's-eye, eye-level medium shot, pull-back reveal, etc.).
9. Maintain a clear educational and demonstration tone throughout.
10. Each scene may include a subtle narration style cue such as:
    - calm instructional voice-over
    - clear step-by-step explanation tone
    - steady professional demonstration narration
    - focused technical commentary tone
11. Do NOT include spoken dialogue between characters.
12. Keep each scene description visually rich but focused on one idea or action.
13. Scenes must follow the exact logical and sequential order of the instructional summary provided.
14. Do NOT collapse multiple distinct steps or topics into one scene. Each major step, warning, or outcome deserves its own scene.
15. Scene count must reflect the depth and length of the summary:
    - Short summary (400–700 words): Generate 8–12 scenes.
    - Medium summary (700–1500 words): Generate 12–20 scenes.
    - Long summary (1500–3000+ words): Generate 20–35 scenes or more as needed.
16. Each scene represents exactly one 8-second video clip. When writing each scene description, mentally pace the action and visuals to confirm they fit naturally within 8 seconds — not rushed, not lingering. Apply these guidelines:
    - A single focused action (adjusting a knob, placing a sample, pouring a liquid) fits one 8-second scene.
    - If a step involves multiple distinct actions or transitions that would feel rushed or incomplete in 8 seconds, split it into two or more separate scenes without hesitation.
    - If a concept or visual is simple and resolved quickly, keep it as one scene — do not pad or stretch.
    - Never compress two full actions into one scene just to reduce scene count.
    - Always prioritize timing accuracy over scene count targets. If the content demands 40 scenes to cover everything at a proper 8-second pace, generate 40 scenes.
    - As a general pacing reference: one calm deliberate action, one camera movement, and one environmental detail = approximately 8 seconds of screen time.
17. Never truncate the summary content to fit a scene limit. Scene count must expand to cover all instructional content completely.

Output Example:

[
  "Wide establishing shot of a clean university biology lab with rows of equipment under soft white fluorescent lighting. A student in a lab coat carefully prepares sample slides at a workstation. Calm instructional voice-over tone.",
  "Macro close-up shot of gloved hands placing a thin tissue sample onto a glass slide under warm task lighting. The motion is slow and deliberate. Clear step-by-step explanation tone.",
  "Medium eye-level shot of the student adjusting the focus knob on a compound microscope. The background shows blurred lab equipment. Steady professional demonstration narration.",
  "Next scene..."
]
      `;

        let lastError: unknown;

        for (let attempt = 1; attempt <= GeminiService.VIDEO_SCRIPT_MAX_RETRIES; attempt++) {
            try {
                const result = await this.openai.responses.create({
                    model: this.sceneModel,
                    // Long summaries ask for many verbose scenes; 8k tokens often truncates mid-JSON → JSON.parse errors.
                    max_output_tokens: 16_384,
                    input: [
                        { role: "system", content: systemInstruction },
                        { role: "user", content },
                    ],
                });

                const rawText = extractOpenAIResponseOutputText(result);

                const cleaned = rawText
                    .replace(/```json\s*/gi, "")
                    .replace(/```\s*/g, "")
                    .trim();

                if (!cleaned) {
                    throw new Error(
                        "Scene model returned empty output (expected a JSON array of scene strings).",
                    );
                }

                let parsed: unknown;
                try {
                    parsed = JSON.parse(cleaned);
                } catch (parseErr) {
                    const likelyTruncated =
                        cleaned.startsWith("[") &&
                        !/\]\s*$/.test(cleaned) &&
                        cleaned.length > 100;
                    throw new Error(
                        likelyTruncated
                            ? "Scene JSON looks truncated (output may have hit max_output_tokens). Try a shorter document summary or raise the token limit."
                            : parseErr instanceof Error
                                ? parseErr.message
                                : String(parseErr),
                    );
                }

                if (!Array.isArray(parsed) || !parsed.every((x) => typeof x === "string")) {
                    throw new Error("OpenAI did not return a JSON array of strings.");
                }

                return parsed;
            } catch (err: any) {
                lastError = err;
                console.warn(
                    `[geminiVideoScript] Attempt ${attempt}/${GeminiService.VIDEO_SCRIPT_MAX_RETRIES} failed: ${err?.message}`,
                );
                if (attempt < GeminiService.VIDEO_SCRIPT_MAX_RETRIES) {
                    await sleep(1000 * attempt);
                }
            }
        }

        throw lastError;
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
                config: { aspectRatio },
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
                    config: { aspectRatio },
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
    async concatClipsFFmpeg(clipPaths: string[], outputPath: string) {
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
    async concatClipsFFmpegReencode(clipPaths: string[], outputPath: string) {
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

    /**
     * One shared visual bible reused for every keyframe so scenes stay stylistically consistent.
     */
    async geminiVisualStyleBrief(content: string, scenes: string[]): Promise<string> {
        const systemInstruction = `You write a single compact "visual style bible" (about 120–200 words) for an educational video series.
        Lock these across all episodes/frames: color palette, lighting approach, realism level, recurring characters or subjects (if any), wardrobe/props consistency, environment tone, and lens/camera language.
        Do not narrate the lesson step-by-step. Do not number scenes. Plain text only, no markdown.`;

        const user = `Original instructional content:\n\n${content}\n\nScene prompts (context only):\n${JSON.stringify(scenes)}`;

        const result: any = await this.ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: [{ role: "user", parts: [{ text: user }] }],
            config: {
                temperature: 0.65,
                maxOutputTokens: 1024,
                systemInstruction,
            },
        });

        const text = result?.text?.trim?.() ?? "";
        if (!text) {
            throw new Error("geminiVisualStyleBrief: empty response from Gemini.");
        }
        console.log("[scene-image-montage] Step complete: visual style brief generated.");
        return text;
    }

    private extractImageBytesFromModelResponse(res: any): { base64: string; mimeType: string } {
        const gi = res?.generatedImages?.[0]?.image;
        if (gi?.imageBytes) {
            return { base64: String(gi.imageBytes), mimeType: gi.mimeType || "image/png" };
        }
        const parts = res?.candidates?.[0]?.content?.parts ?? [];
        for (const p of parts) {
            if (p?.inlineData?.data) {
                return { base64: String(p.inlineData.data), mimeType: p.inlineData.mimeType || "image/png" };
            }
        }
        throw new Error("Image model returned no image bytes.");
    }

    private async generateSceneKeyframeFlashImage(args: {
        sceneIndex: number;
        total: number;
        scenePrompt: string;
        styleBrief: string;
        referenceBase64?: string;
        referenceMime?: string;
        aspectRatio: "9:16" | "16:9";
    }): Promise<{ base64: string; mimeType: string }> {
        const { sceneIndex, total, scenePrompt, styleBrief, referenceBase64, referenceMime, aspectRatio } = args;

        const header = `Educational video keyframe, photorealistic cinematic still. Scene ${sceneIndex + 1} of ${total}.

        STYLE BIBLE (must stay consistent across every image in this series):
        ${styleBrief}

        SHOT / SCENE:
        ${scenePrompt}

        Single frame only. No subtitles, no on-screen text, no split panels.`;

        const parts: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }> = [];
        if (referenceBase64) {
            parts.push({
                inlineData: {
                    mimeType: referenceMime || "image/png",
                    data: referenceBase64,
                },
            });
            parts.push({
                text: `The attached image is the series visual anchor. Match its characters, wardrobe, props, palette, lighting mood, and production design. Create a NEW keyframe for the shot below (new composition; same world).

${header}`,
            });
        } else {
            parts.push({ text: header });
        }

        const res: any = await this.ai.models.generateContent({
            model: FLASH_IMAGE_MODEL,
            contents: [{ role: "user", parts }],
            config: {
                responseModalities: [Modality.IMAGE],
                imageConfig: { aspectRatio },
            },
        });

        const out = this.extractImageBytesFromModelResponse(res);
        console.log(
            `[scene-image-montage] Step complete: flash keyframe generated (scene ${args.sceneIndex + 1}/${args.total}).`,
        );
        return out;
    }

    private async generateSceneKeyframeImagen(
        prompt: string,
        aspectRatio: "9:16" | "16:9",
        sceneMeta?: { index1Based: number; total: number },
    ): Promise<{ base64: string; mimeType: string }> {
        const res = await this.ai.models.generateImages({
            model: IMAGEN_GENERATE_MODEL,
            prompt,
            config: {
                numberOfImages: 1,
                aspectRatio,
                outputMimeType: "image/png",
            },
        });

        const first = res?.generatedImages?.[0];
        const img = first?.image;
        if (!img?.imageBytes) {
            const rai = first?.raiFilteredReason;
            throw new Error(`Imagen did not return an image.${rai ? ` RAI: ${rai}` : ""}`);
        }
        const out = { base64: String(img.imageBytes), mimeType: img.mimeType || "image/png" };
        if (sceneMeta) {
            console.log(
                `[scene-image-montage] Step complete: Imagen keyframe generated (scene ${sceneMeta.index1Based}/${sceneMeta.total}).`,
            );
        } else {
            console.log("[scene-image-montage] Step complete: Imagen keyframe generated.");
        }
        return out;
    }

    private async pollVideoOperationUntilDone(operation: any, label: string) {
        const maxAttempts = 120;
        let attempt = 0;
        let op = operation;
        while (!op.done) {
            attempt += 1;
            if (attempt > maxAttempts) {
                throw new Error(`${label}: Veo generation timed out while polling operation status.`);
            }
            await sleep(10_000);
            op = await this.ai.operations.getVideosOperation({ operation: op });
        }
        console.log(`[scene-image-montage] Step complete: Veo operation finished (${label}).`);
        return op;
    }

    /** Veo LRO can be `done` with `error` set, or with no downloadable video — log clearly before throwing. */
    private assertVeoOperationHasDownloadableVideo(operation: any, label: string): any {
        if (operation?.error != null) {
            const detail =
                typeof operation.error === "object"
                    ? JSON.stringify(operation.error)
                    : String(operation.error);
            console.error(`[scene-image-montage] Veo operation returned error (${label}):`, detail);
            throw new Error(`Veo operation failed (${label}): ${detail}`);
        }
        const generated = operation?.response?.generatedVideos?.[0];
        const file = generated?.video;
        if (!file) {
            const snap = {
                hasResponse: !!operation?.response,
                generatedVideosLength: operation?.response?.generatedVideos?.length ?? 0,
                firstEntryKeys: generated && typeof generated === "object" ? Object.keys(generated) : [],
            };
            console.error(
                `[scene-image-montage] Veo finished but no video file in response (${label}). Snapshot:`,
                snap,
            );
            throw new Error(
                `No generated video in response for ${label}. Check server logs for snapshot; Veo may have filtered the clip or returned an empty payload.`,
            );
        }
        return file;
    }

    private async downloadVeoVideoToPath(file: any, downloadPath: string, label: string) {
        const attempts = 3;
        let lastErr: unknown;
        for (let a = 1; a <= attempts; a++) {
            try {
                await this.ai.files.download({ file, downloadPath });
                return;
            } catch (err: any) {
                lastErr = err;
                console.error(
                    `[scene-image-montage] files.download failed (${label}, attempt ${a}/${attempts}):`,
                    err?.message ?? err,
                    err?.stack ?? "",
                );
                if (a < attempts) {
                    await sleep(2000 * a);
                }
            }
        }
        throw lastErr;
    }

    /** Finished LRO `error` payload from Veo (e.g. `{ code: 13, message: "..." }`). */
    private isRetryableVeoOperationError(error: unknown): boolean {
        if (error == null || typeof error !== "object") {
            return false;
        }
        const e = error as { code?: number; message?: string };
        const code = e.code;
        const msg = (e.message ?? "").toLowerCase();
        if (code === 13 || code === 14 || code === 8) {
            return true;
        }
        return /try again|internal server|unavailable|temporarily|timeout|resource exhausted/.test(msg);
    }

    private isRetryableVeoSdkError(err: unknown): boolean {
        const msg = String((err as { message?: string })?.message ?? err).toLowerCase();
        return /try again|internal server|unavailable|temporarily|code.*13|\b500\b|deadline|timeout|econnreset|socket/.test(
            msg,
        );
    }

    /**
     * One montage clip: start Veo image-to-video, poll, validate, download.
     * Retries the whole clip on transient Gemini INTERNAL / unavailable style failures.
     */
    private async runMontageVeoClipWithRetries(args: {
        clipIndex1Based: number;
        aspectRatio: "9:16" | "16:9";
        imageBytes: string;
        mimeType: string;
        videoPrompt: string;
        downloadPath: string;
        /** Optional log label override (e.g. `scene 3 clip 2/4`). Defaults to `clip <n>`. */
        label?: string;
    }): Promise<void> {
        const label = args.label ?? `clip ${args.clipIndex1Based}`;
        let lastFailure: unknown;

        for (let attempt = 1; attempt <= VEO_MONTAGE_CLIP_MAX_ATTEMPTS; attempt++) {
            try {
                let operation = await this.ai.models.generateVideos({
                    model: VEO_MODEL,
                    prompt: args.videoPrompt,
                    image: { imageBytes: args.imageBytes, mimeType: args.mimeType },
                    config: { aspectRatio: args.aspectRatio, resolution: "720p", numberOfVideos: 1 },
                });

                operation = await this.pollVideoOperationUntilDone(operation, label);

                if (operation?.error != null) {
                    const retryOp =
                        this.isRetryableVeoOperationError(operation.error) &&
                        attempt < VEO_MONTAGE_CLIP_MAX_ATTEMPTS;
                    if (retryOp) {
                        const waitMs = VEO_MONTAGE_CLIP_RETRY_BASE_MS * attempt;
                        console.warn(
                            `[scene-image-montage] ${label}: retryable Veo operation error (attempt ${attempt}/${VEO_MONTAGE_CLIP_MAX_ATTEMPTS}), waiting ${waitMs}ms then re-running clip:`,
                            operation.error,
                        );
                        await sleep(waitMs);
                        lastFailure = operation.error;
                        continue;
                    }
                }

                const videoFile = this.assertVeoOperationHasDownloadableVideo(operation, label);
                await this.downloadVeoVideoToPath(videoFile, args.downloadPath, label);
                if (attempt > 1) {
                    console.log(
                        `[scene-image-montage] ${label}: succeeded after ${attempt} attempt(s).`,
                    );
                }
                return;
            } catch (err: unknown) {
                lastFailure = err;
                const errMsg = (err as { message?: string })?.message ?? String(err);
                const errStack = (err as { stack?: string })?.stack ?? "";
                const errCode = (err as { code?: unknown })?.code;
                const errStatus = (err as { status?: unknown })?.status;
                const errStatusCode = (err as { statusCode?: unknown })?.statusCode;

                console.error(
                    `[scene-image-montage] ${label}: generateVideos/poll/download threw (attempt ${attempt}/${VEO_MONTAGE_CLIP_MAX_ATTEMPTS}):`,
                    {
                        message: errMsg,
                        code: errCode,
                        status: errStatus,
                        statusCode: errStatusCode,
                        stack: errStack,
                        rawError: err,
                    },
                );

                const retry =
                    attempt < VEO_MONTAGE_CLIP_MAX_ATTEMPTS && this.isRetryableVeoSdkError(err);
                if (retry) {
                    const waitMs = VEO_MONTAGE_CLIP_RETRY_BASE_MS * attempt;
                    console.warn(
                        `[scene-image-montage] ${label}: will retry in ${waitMs}ms...`,
                    );
                    await sleep(waitMs);
                    continue;
                }
                console.error(
                    `[scene-image-montage] ${label}: error is NOT retryable — propagating.`,
                );
                throw err;
            }
        }

        const detail =
            lastFailure != null && typeof lastFailure === "object" && "message" in (lastFailure as object)
                ? (lastFailure as { message: string }).message
                : String(lastFailure);
        throw new Error(
            `${label}: Veo failed after ${VEO_MONTAGE_CLIP_MAX_ATTEMPTS} attempts. Last error: ${detail}`,
        );
    }

    private async runStepWithRetries<T>(
        label: string,
        fn: () => Promise<T>,
        maxRetries = PIPELINE_STEP_MAX_RETRIES,
    ): Promise<T> {
        let lastError: unknown;
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                return await fn();
            } catch (err) {
                lastError = err;
                const msg =
                    err instanceof Error ? err.message : typeof err === "string" ? err : JSON.stringify(err);
                console.warn(
                    `[scene-image-montage] ${label} failed (attempt ${attempt}/${maxRetries}): ${msg}`,
                );
                if (attempt < maxRetries) {
                    await sleep(PIPELINE_STEP_RETRY_BASE_MS * attempt);
                }
            }
        }
        throw lastError;
    }

    private async fileExists(filePath: string): Promise<boolean> {
        try {
            await fs.access(filePath);
            return true;
        } catch {
            return false;
        }
    }

    private async collectExistingFiles(filePaths: string[]): Promise<string[]> {
        const existing: string[] = [];
        for (const p of filePaths) {
            if (await this.fileExists(p)) {
                existing.push(p);
            }
        }
        return existing;
    }

    private async clipFileByteSize(filePath: string): Promise<number> {
        try {
            const st = await fs.stat(filePath);
            return st.size;
        } catch {
            return 0;
        }
    }

    private normalizeResumeMontageId(raw?: string): string | undefined {
        const trimmed = (raw ?? "").trim();
        if (!trimmed) return undefined;
        if (!/^[A-Za-z0-9_-]+$/.test(trimmed)) {
            throw new Error("resumeMontageId contains invalid characters.");
        }
        return trimmed;
    }

    private normalizeResumeFromScene(value?: number): number {
        if (value == null) return 1;
        if (!Number.isFinite(value)) return 1;
        return Math.max(1, Math.floor(value));
    }

    private async resolveExistingSceneImagePath(imagesDir: string, sceneIndex1Based: number): Promise<string | null> {
        const stem = `scene_${String(sceneIndex1Based).padStart(2, "0")}`;
        const candidates = [`${stem}.png`, `${stem}.jpg`, `${stem}.jpeg`].map((n) => path.join(imagesDir, n));
        for (const p of candidates) {
            if (await this.fileExists(p)) {
                const size = await this.clipFileByteSize(p);
                if (size > 0) return p;
            }
        }
        return null;
    }

    /**
     * All finished scene clips in the montage workspace (clip_01.mp4, clip_02.mp4, …), scene order.
     * Scans the directory so partial stitch includes every clip that actually landed on disk, then fills
     * gaps from the manifest paths (same paths in normal runs).
     */
    private async collectMontageWorkspaceClips(montageDir: string, clipFiles: string[]): Promise<string[]> {
        const byScene = new Map<number, string>();

        const tryAdd = async (p: string) => {
            const base = path.basename(p);
            const m = base.match(/^clip_(\d+)\.mp4$/i);
            if (!m) return;
            const idx = parseInt(m[1], 10);
            if (!Number.isFinite(idx) || idx < 1) return;
            const abs = path.resolve(p);
            if (!(await this.fileExists(abs))) return;
            if ((await this.clipFileByteSize(abs)) === 0) return;
            if (!byScene.has(idx)) {
                byScene.set(idx, abs);
            }
        };

        try {
            const names = await fs.readdir(montageDir);
            const re = /^clip_(\d+)\.mp4$/i;
            const hits: { n: number; full: string }[] = [];
            for (const name of names) {
                const match = name.match(re);
                if (match) {
                    hits.push({ n: parseInt(match[1], 10), full: path.join(montageDir, name) });
                }
            }
            hits.sort((a, b) => a.n - b.n);
            for (const h of hits) {
                await tryAdd(h.full);
            }
        } catch {
            // montageDir missing or unreadable — fall through to manifest paths only
        }

        for (const p of clipFiles) {
            await tryAdd(p);
        }

        return [...byScene.keys()].sort((a, b) => a - b).map((k) => byScene.get(k)!);
    }

    private async stitchSuccessfulClips(args: {
        montageDir: string;
        clipFiles: string[];
        outDir: string;
        montageId: string;
    }): Promise<{
        stitchedClipCount: number;
        partialCombinedFilePath: string | null;
        partialCombinedFileName: string | null;
    }> {
        const existingClipFiles = await this.collectMontageWorkspaceClips(args.montageDir, args.clipFiles);
        if (!existingClipFiles.length) {
            return {
                stitchedClipCount: 0,
                partialCombinedFilePath: null,
                partialCombinedFileName: null,
            };
        }

        console.log(
            `[scene-image-montage] partial stitch: combining ${existingClipFiles.length} clip file(s) from workspace.`,
        );

        const partialCombinedFileName = `${args.montageId}_partial_${Date.now()}.mp4`;
        const partialCombinedFilePath = path.join(args.outDir, partialCombinedFileName);

        if (existingClipFiles.length === 1) {
            await fs.copyFile(existingClipFiles[0], partialCombinedFilePath);
            return {
                stitchedClipCount: 1,
                partialCombinedFilePath,
                partialCombinedFileName,
            };
        }

        try {
            await this.concatClipsFFmpeg(existingClipFiles, partialCombinedFilePath);
        } catch {
            await this.concatClipsFFmpegReencode(existingClipFiles, partialCombinedFilePath);
        }

        return {
            stitchedClipCount: existingClipFiles.length,
            partialCombinedFilePath,
            partialCombinedFileName,
        };
    }

    /**
     * content → scenes → shared style brief → keyframe images (folder) → image-to-video per scene → concat (same as montage).
     */
    async generateSceneImageMontageFromContent(
        content: string,
        aspectRatio: "9:16" | "16:9" = "9:16",
        resume: SceneImageMontageResumeOptions = {},
    ) {
        console.log("[scene-image-montage] Pipeline start: scene-with-images montage.");
        // Preflight: ffmpeg is only used at the final stitch step, but the whole pipeline is
        // worthless (and expensive) if we only discover it's missing AFTER generating every Veo clip.
        await this.assertFfmpegAvailable();
        console.log("[scene-image-montage] Preflight complete: ffmpeg is available.");
        const outDir = path.resolve(process.cwd(), "generated-document-to-video");
        await fs.mkdir(outDir, { recursive: true });
        console.log("[scene-image-montage] Step complete: output directory ready.", outDir);

        const resumeMontageId = this.normalizeResumeMontageId(resume.resumeMontageId);
        const resumeFromScene = this.normalizeResumeFromScene(resume.resumeFromScene);
        const montageId = resumeMontageId ?? `montage_img_${Date.now()}`;
        const montageDir = path.join(outDir, montageId);
        const imagesDir = path.join(montageDir, "images");
        await fs.mkdir(imagesDir, { recursive: true });
        console.log("[scene-image-montage] Step complete: montage workspace ready.", {
            montageDir,
            imagesDir,
            resumeFromScene,
            isResume: !!resumeMontageId,
        });

        let failedStep = "initialization";
        let scenes: string[] = [];
        let styleBrief = "";
        const imagePaths: string[] = [];
        let clipFiles: string[] = [];
        let anchorBase64: string | undefined;
        let anchorMime = "image/png";

        try {
            failedStep = "scene generation";
            scenes = await this.runStepWithRetries("scene generation", () => this.geminiVideoScript(content));
            console.log("scenes presently here is ", scenes);
            if (!scenes.length) {
                throw new Error("No scenes produced from content.");
            }
            if (resumeFromScene > scenes.length) {
                throw new Error(
                    `resumeFromScene (${resumeFromScene}) is greater than generated scene count (${scenes.length}).`,
                );
            }
            console.log(
                `[scene-image-montage] Step complete: video script / scenes generated (count=${scenes.length}).`,
            );

            failedStep = "style brief generation";
            styleBrief = await this.runStepWithRetries("style brief generation", () =>
                this.geminiVisualStyleBrief(content, scenes),
            );
            console.log(
                `[scene-image-montage] Step complete: style brief ready (length=${styleBrief.length} chars).`,
            );

            clipFiles = scenes.map((_, i) =>
                path.join(montageDir, `clip_${String(i + 1).padStart(2, "0")}.mp4`),
            );

            for (let i = 0; i < scenes.length; i++) {
                imagePaths[i] = path.join(imagesDir, `scene_${String(i + 1).padStart(2, "0")}.png`);
            }
            for (let i = 0; i < scenes.length; i++) {
                const existing = await this.resolveExistingSceneImagePath(imagesDir, i + 1);
                if (existing) {
                    imagePaths[i] = existing;
                }
            }

            const anchorPath = imagePaths[0];
            if (anchorPath && (await this.fileExists(anchorPath))) {
                const anchorBuf = await fs.readFile(anchorPath);
                anchorBase64 = anchorBuf.toString("base64");
                anchorMime = anchorPath.toLowerCase().endsWith(".jpg") || anchorPath.toLowerCase().endsWith(".jpeg")
                    ? "image/jpeg"
                    : "image/png";
                console.log("[scene-image-montage] Reusing first scene image as visual anchor.");
            }

            console.log(`[scene-image-montage] Phase: generating ${scenes.length} keyframe images...`);
            for (let i = resumeFromScene - 1; i < scenes.length; i++) {
                const keyframeStep = `keyframe ${i + 1}/${scenes.length}`;
                failedStep = keyframeStep;

                const existingImagePath = await this.resolveExistingSceneImagePath(imagesDir, i + 1);
                if (existingImagePath) {
                    imagePaths[i] = existingImagePath;
                    console.log(
                        `[scene-image-montage] ${keyframeStep}: existing image found, skipping generation -> ${existingImagePath}`,
                    );
                    continue;
                }

                const packed = await this.runStepWithRetries(keyframeStep, async () => {
                    try {
                        return await this.generateSceneKeyframeFlashImage({
                            sceneIndex: i,
                            total: scenes.length,
                            scenePrompt: scenes[i].trim(),
                            styleBrief,
                            referenceBase64: anchorBase64,
                            referenceMime: anchorMime,
                            aspectRatio,
                        });
                    } catch (err: any) {
                        console.warn(
                            `[scene-image-montage] flash-image failed scene ${i + 1}, falling back to Imagen:`,
                            err?.message ?? err,
                        );
                        const combined = `STYLE BIBLE (keep consistent):\n${styleBrief}\n\nScene ${i + 1} of ${scenes.length} — cinematic educational keyframe, no text overlays:\n${scenes[i].trim()}`;
                        return await this.generateSceneKeyframeImagen(combined, aspectRatio, {
                            index1Based: i + 1,
                            total: scenes.length,
                        });
                    }
                });

                if (anchorBase64 === undefined) {
                    anchorBase64 = packed.base64;
                    anchorMime = packed.mimeType;
                    console.log("[scene-image-montage] Step complete: visual anchor set from first keyframe.");
                }

                const ext =
                    packed.mimeType.includes("jpeg") || packed.mimeType.includes("jpg") ? "jpg" : "png";
                const imagePath = path.join(imagesDir, `scene_${String(i + 1).padStart(2, "0")}.${ext}`);
                await fs.writeFile(imagePath, Buffer.from(packed.base64, "base64"));
                imagePaths[i] = imagePath;
                console.log(
                    `[scene-image-montage] Step complete: keyframe ${i + 1}/${scenes.length} saved to disk -> ${imagePath}`,
                );
            }
            console.log("[scene-image-montage] Phase complete: all keyframe images written.");

            const ar = aspectRatio as "9:16" | "16:9";
            console.log(`[scene-image-montage] Phase: generating ${scenes.length} Veo clips from keyframes...`);
            for (let i = resumeFromScene - 1; i < scenes.length; i++) {
                const existingClipBytes = await this.clipFileByteSize(clipFiles[i]);
                if (existingClipBytes > 0) {
                    console.log(
                        `[scene-image-montage] video clip ${i + 1}/${scenes.length}: existing clip found, skipping Veo generation -> ${clipFiles[i]}`,
                    );
                    continue;
                }
                if (!imagePaths[i] || !(await this.fileExists(imagePaths[i]))) {
                    throw new Error(
                        `Missing keyframe image for scene ${i + 1}. Expected an image before generating clip.`,
                    );
                }
                const imageBuf = await fs.readFile(imagePaths[i]);
                const imageBytes = imageBuf.toString("base64");
                const mimeType = imagePaths[i].toLowerCase().endsWith(".jpg")
                    ? "image/jpeg"
                    : "image/png";

                const videoPrompt = `Scene ${i + 1} of ${scenes.length}. Animate this keyframe with cinematic motion; keep subjects and style faithful to the image. ${scenes[i].trim()}`;

                failedStep = `video clip ${i + 1}/${scenes.length}`;
                console.log(`[scene-image-montage] Starting Veo clip ${i + 1}/${scenes.length}...`);
                await this.runMontageVeoClipWithRetries({
                    clipIndex1Based: i + 1,
                    aspectRatio: ar,
                    imageBytes,
                    mimeType,
                    videoPrompt,
                    downloadPath: clipFiles[i],
                });
                console.log(
                    `[scene-image-montage] Step complete: clip ${i + 1}/${scenes.length} downloaded -> ${clipFiles[i]}`,
                );
            }
            console.log("[scene-image-montage] Phase complete: all Veo clips saved.");

            const combinedFileName = `${montageId}.mp4`;
            const combinedFilePath = path.join(outDir, combinedFileName);

            failedStep = "final stitching";
            console.log("[scene-image-montage] Phase: stitching clips with FFmpeg...");
            try {
                await this.concatClipsFFmpeg(clipFiles, combinedFilePath);
                console.log("[scene-image-montage] Step complete: FFmpeg concat (stream copy) succeeded.");
            } catch {
                console.log("[scene-image-montage] Stream copy concat failed; falling back to re-encode concat.");
                await this.concatClipsFFmpegReencode(clipFiles, combinedFilePath);
                console.log("[scene-image-montage] Step complete: FFmpeg re-encode concat succeeded.");
            }

            console.log("[scene-image-montage] Pipeline complete: final montage ->", combinedFilePath);

            return {
                combinedFilePath,
                combinedFileName,
                clipFiles,
                imagePaths,
                imagesDir,
                montageDir,
                montageId,
                scenes,
                styleBrief,
            };
        } catch (err: any) {
            const partial = await this.stitchSuccessfulClips({
                montageDir,
                clipFiles,
                outDir,
                montageId,
            });
            const originalErrorMessage =
                err instanceof Error ? err.message : typeof err === "string" ? err : JSON.stringify(err);
            const details: SceneImageMontageFailureDetails = {
                step: failedStep,
                montageId,
                montageDir,
                imagesDir,
                retriesPerStep: PIPELINE_STEP_MAX_RETRIES,
                successfulImageCount: imagePaths.length,
                successfulClipCount: partial.stitchedClipCount,
                partialCombinedFilePath: partial.partialCombinedFilePath,
                partialCombinedFileName: partial.partialCombinedFileName,
                originalErrorMessage,
            };
            throw new SceneImageMontageError(
                `[scene-image-montage] failed at ${failedStep} after retries: ${originalErrorMessage}`,
                details,
            );
        }
    }

    // ===========================================================================================
    // Duration-aware document-to-video pipeline
    // -------------------------------------------------------------------------------------------
    // Implements APPROACH.md: each scene now carries its narration script and a derived audio
    // duration. `clipsNeeded = ceil(audioDurationSeconds / 8)` so a long scene spans multiple
    // chained Veo clips (last frame of clip i seeds clip i+1) and is then concat-and-trimmed back
    // to the exact narration duration before the final video-wide concat.
    // ===========================================================================================

    /**
     * Same role as {@link geminiVideoScript} but returns paired visual description + narration
     * script per scene. The narration script length is what drives `clipsNeeded` downstream, so
     * it must be written to fully cover the topic (no artificial padding/truncation).
     */
    async geminiVideoScriptWithNarration(
        content: string,
    ): Promise<{ description: string; script: string }[]> {
        const systemInstruction = `
You are an expert cinematic educational scene designer AND narration writer.

Your task is to convert a detailed instructional summary into structured scene-based plans for educational/demonstration videos. Each plan contains a cinematic visual description AND the narrator voice-over script for that scene.

Output Requirements:
1. Return ONLY a valid JSON array of objects. No markdown, no commentary, no code fences.
2. Each object must have exactly two string fields: "description" and "script".
3. "description" — a single cinematic, photorealistic visual scene description suitable for AI video generation. Include environment details, lighting, and a clear camera movement / shot type. Use a steady educational/demonstration tone.
4. "script" — the narrator voice-over text spoken while this visual plays. Plain prose only. No SSML, no markdown, no quotation marks, no character dialogue, no stage directions, no scene labels, no scene numbers.
5. Each scene covers ONE coherent topic, step, concept, warning, or outcome from the summary, in the same order the reader encounters it. Do not collapse multiple distinct steps into one scene.
6. The script length determines the on-screen duration of the scene at roughly ${NARRATION_WORDS_PER_SECOND} spoken words per second. Write enough script to fully cover the topic — do not artificially shorten or pad.
   - A simple action or transition is typically 8–16 seconds (~20–40 words).
   - A standard explained step is typically 16–32 seconds (~40–80 words).
   - A complex concept, warning, or outcome may be 24+ seconds (~60+ words or more as needed).
7. Scene count must reflect the depth of the summary. Do not truncate. Short summaries (~400–700 words) ≈ 8–12 scenes. Medium (~700–1500 words) ≈ 12–20 scenes. Long (1500–3000+ words) ≈ 20–35+ scenes.
8. Maintain consistent terminology with the source content.
9. Never include speaker tags, scene numbers, or headings inside either field.

Output Example:
[
  {
    "description": "Wide establishing shot of a clean university biology lab with rows of equipment under soft white fluorescent lighting. A student in a lab coat carefully prepares sample slides at a workstation. Slow zoom in. Calm instructional voice-over tone.",
    "script": "Welcome to the cell biology lab. Today we will walk through the steps of preparing tissue samples for microscopic observation, starting with workstation setup and the equipment you will use throughout this demonstration."
  },
  {
    "description": "Macro close-up of gloved hands placing a thin tissue sample onto a glass slide under warm task lighting. The motion is slow and deliberate. Clear step-by-step explanation tone.",
    "script": "Carefully transfer the tissue sample to a clean glass slide using forceps. Keep the sample flat and avoid touching the surface with your fingers, as oils and contaminants can affect staining results downstream."
  }
]
        `.trim();

        let lastError: unknown;

        for (let attempt = 1; attempt <= GeminiService.VIDEO_SCRIPT_MAX_RETRIES; attempt++) {
            try {
                const result = await this.openai.responses.create({
                    model: this.sceneModel,
                    // Long summaries with narration scripts can be verbose; keep token ceiling generous.
                    max_output_tokens: 16_384,
                    input: [
                        { role: "system", content: systemInstruction },
                        { role: "user", content },
                    ],
                });

                const rawText = extractOpenAIResponseOutputText(result);

                const cleaned = rawText
                    .replace(/```json\s*/gi, "")
                    .replace(/```\s*/g, "")
                    .trim();

                if (!cleaned) {
                    throw new Error(
                        "Scene+script model returned empty output (expected JSON array of {description, script}).",
                    );
                }

                let parsed: unknown;
                try {
                    parsed = JSON.parse(cleaned);
                } catch (parseErr) {
                    const likelyTruncated =
                        cleaned.startsWith("[") &&
                        !/]\s*$/.test(cleaned) &&
                        cleaned.length > 100;
                    throw new Error(
                        likelyTruncated
                            ? "Scene+script JSON looks truncated (output may have hit max_output_tokens). Try a shorter summary or raise the token limit."
                            : parseErr instanceof Error
                                ? parseErr.message
                                : String(parseErr),
                    );
                }

                if (
                    !Array.isArray(parsed) ||
                    !parsed.every(
                        (x) =>
                            x != null &&
                            typeof x === "object" &&
                            typeof (x as { description?: unknown }).description === "string" &&
                            typeof (x as { script?: unknown }).script === "string",
                    )
                ) {
                    throw new Error(
                        "Scene model did not return a JSON array of {description, script} string objects.",
                    );
                }

                const out = (parsed as Array<{ description: string; script: string }>)
                    .map((s) => ({
                        description: s.description.trim(),
                        script: s.script.trim(),
                    }))
                    .filter((s) => s.description && s.script);

                if (!out.length) {
                    throw new Error("Scene model returned only empty entries.");
                }

                return out;
            } catch (err: any) {
                lastError = err;
                console.warn(
                    `[geminiVideoScriptWithNarration] Attempt ${attempt}/${GeminiService.VIDEO_SCRIPT_MAX_RETRIES} failed: ${err?.message}`,
                );
                if (attempt < GeminiService.VIDEO_SCRIPT_MAX_RETRIES) {
                    await sleep(1000 * attempt);
                }
            }
        }

        throw lastError;
    }

    /**
     * Public scene+narration plan: runs `geminiVideoScriptWithNarration` (with the same retry
     * wrapper the orchestrator uses) and converts the result into `PlannedScene[]` with audio
     * durations and `clipsNeeded` already filled in.
     *
     * Used by the duration-aware orchestrator (when no resumeable plan exists on disk) AND by
     * the `/estimate-duration` endpoint, so the estimate stays exactly in sync with what the
     * real `/document-to-video` run will produce — same scene count, same per-scene narration
     * duration, same Veo clip count.
     */
    async planDurationAwareScenesFromContent(content: string): Promise<PlannedScene[]> {
        const rawScenes = await this.runStepWithRetries("scene generation", () =>
            this.geminiVideoScriptWithNarration(content),
        );
        if (!rawScenes.length) {
            throw new Error("No scenes produced from content.");
        }
        return this.planScenes(rawScenes);
    }

    /** Cheap heuristic: ~2.5 words / second of narration (matches APPROACH.md). */
    private estimateDurationFromScript(script: string): number {
        const wordCount = script.trim().split(/\s+/).filter(Boolean).length;
        if (wordCount === 0) return VEO_CLIP_DURATION_SECONDS;
        return Math.max(1, Math.ceil(wordCount / NARRATION_WORDS_PER_SECOND));
    }

    /** Attaches scene id, audio duration, and `clipsNeeded` (ceil(audio / 8)) to each raw scene. */
    private planScenes(
        rawScenes: { description: string; script: string }[],
    ): PlannedScene[] {
        return rawScenes.map((raw, i) => {
            const audioDurationSeconds = this.estimateDurationFromScript(raw.script);
            const clipsNeeded = Math.max(
                1,
                Math.ceil(audioDurationSeconds / VEO_CLIP_DURATION_SECONDS),
            );
            return {
                id: `scene_${String(i + 1).padStart(2, "0")}`,
                sceneIndex: i,
                description: raw.description,
                script: raw.script,
                audioDurationSeconds,
                clipsNeeded,
                referenceImages: [],
            };
        });
    }

    /**
     * Frame labels per APPROACH.md frame-generator:
     *   1 clip  → ["establishing"]
     *   2 clips → ["start", "end"]
     *   3 clips → ["start", "mid-1", "end"]
     *   N clips → ["start", "mid-1", ..., "mid-(N-2)", "end"]
     */
    private frameLabelsForClipCount(clipsNeeded: number): string[] {
        if (clipsNeeded <= 1) return ["establishing"];
        const labels: string[] = ["start"];
        for (let i = 1; i < clipsNeeded - 1; i++) {
            labels.push(`mid-${i}`);
        }
        labels.push("end");
        return labels;
    }

    /** `scene_NN_frame_FF.{png,jpg,jpeg}` lookup for resume reuse. */
    private async resolveExistingFramePath(
        imagesDir: string,
        sceneIndex1Based: number,
        frameIndex1Based: number,
    ): Promise<string | null> {
        const stem = `scene_${String(sceneIndex1Based).padStart(2, "0")}_frame_${String(
            frameIndex1Based,
        ).padStart(2, "0")}`;
        const candidates = [`${stem}.png`, `${stem}.jpg`, `${stem}.jpeg`].map((n) =>
            path.join(imagesDir, n),
        );
        for (const p of candidates) {
            if (await this.fileExists(p)) {
                if ((await this.clipFileByteSize(p)) > 0) return p;
            }
        }
        return null;
    }

    /**
     * `ffmpeg -sseof -0.1 -i clip.mp4 -frames:v 1 lastframe.jpg`
     * Used to chain the last frame of clip i into clip i+1's start image.
     */
    private async extractLastFrame(clipPath: string, lastFramePath: string): Promise<string> {
        await new Promise<void>((resolve, reject) => {
            const args = [
                "-y",
                "-sseof", "-0.1",
                "-i", clipPath,
                "-frames:v", "1",
                "-q:v", "2",
                lastFramePath,
            ];
            const ff = spawn("ffmpeg", args, { stdio: "pipe" });
            let stderr = "";
            ff.stderr.on("data", (d) => (stderr += d.toString()));
            ff.on("error", (err) => reject(err));
            ff.on("close", (code) => {
                if (code === 0) return resolve();
                reject(new Error(`ffmpeg extractLastFrame failed (code=${code}). Details:\n${stderr}`));
            });
        });
        return lastFramePath;
    }

    /** Returns container duration in seconds, or null if ffprobe is missing or fails. */
    private async probeVideoDurationSeconds(videoPath: string): Promise<number | null> {
        return await new Promise((resolve) => {
            const pr = spawn("ffprobe", [
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                videoPath,
            ]);
            let out = "";
            let err = "";
            pr.stdout?.on("data", (d: Buffer) => (out += d.toString()));
            pr.stderr?.on("data", (d: Buffer) => (err += d.toString()));
            pr.on("error", () => resolve(null));
            pr.on("close", (code) => {
                if (code !== 0) {
                    if (err.trim()) console.warn(`[duration-aware-video] ffprobe failed: ${err.trim()}`);
                    return resolve(null);
                }
                const n = parseFloat(out.trim());
                resolve(Number.isFinite(n) && n > 0 ? n : null);
            });
        });
    }

    /**
     * Per-scene stitch: concatenate Veo clips at their generated length. We intentionally do not
     * trim to `audioDurationSeconds` here — the script-based estimate can be shorter than real
     * narration on the clips, which was cutting the tail (last words) off the stitched scene.
     */
    private async stitchSceneClips(clipPaths: string[], outputPath: string): Promise<void> {
        if (clipPaths.length === 1) {
            await fs.copyFile(clipPaths[0], outputPath);
            return;
        }
        try {
            await this.concatClipsFFmpeg(clipPaths, outputPath);
        } catch {
            await this.concatClipsFFmpegReencode(clipPaths, outputPath);
        }
    }

    /**
     * Partial-recovery concat for the duration-aware pipeline: stitches whatever per-scene videos
     * have already landed on disk so callers get something playable when an individual scene fails.
     */
    private async stitchSuccessfulScenes(args: {
        stitchedScenes: StitchedScene[];
        outDir: string;
        montageId: string;
    }): Promise<{
        stitchedSceneCount: number;
        partialCombinedFilePath: string | null;
        partialCombinedFileName: string | null;
    }> {
        if (!args.stitchedScenes.length) {
            return {
                stitchedSceneCount: 0,
                partialCombinedFilePath: null,
                partialCombinedFileName: null,
            };
        }

        const sources: string[] = [];
        for (const s of args.stitchedScenes) {
            if (await this.fileExists(s.videoPath)) {
                if ((await this.clipFileByteSize(s.videoPath)) > 0) {
                    sources.push(s.videoPath);
                }
            }
        }
        if (!sources.length) {
            return {
                stitchedSceneCount: 0,
                partialCombinedFilePath: null,
                partialCombinedFileName: null,
            };
        }

        console.log(
            `[duration-aware-video] partial stitch: combining ${sources.length} stitched scene video(s).`,
        );

        const partialFileName = `${args.montageId}_partial_${Date.now()}.mp4`;
        const partialPath = path.join(args.outDir, partialFileName);

        if (sources.length === 1) {
            await fs.copyFile(sources[0], partialPath);
            return {
                stitchedSceneCount: 1,
                partialCombinedFilePath: partialPath,
                partialCombinedFileName: partialFileName,
            };
        }

        try {
            await this.concatClipsFFmpeg(sources, partialPath);
        } catch {
            await this.concatClipsFFmpegReencode(sources, partialPath);
        }

        return {
            stitchedSceneCount: sources.length,
            partialCombinedFilePath: partialPath,
            partialCombinedFileName: partialFileName,
        };
    }

    /**
     * Try to load a previously planned scene set from `<montageDir>/scenes.json`. Used when
     * resuming so the per-scene `clipsNeeded` and file paths stay stable across runs (otherwise a
     * regenerated plan from the LLM could change clip counts mid-resume).
     */
    private async loadPlannedScenesFromDisk(montageDir: string): Promise<PlannedScene[] | null> {
        const planPath = path.join(montageDir, "scenes.json");
        if (!(await this.fileExists(planPath))) return null;
        try {
            const raw = await fs.readFile(planPath, "utf8");
            const parsed = JSON.parse(raw);
            if (
                Array.isArray(parsed) &&
                parsed.every(
                    (x) =>
                        x != null &&
                        typeof x === "object" &&
                        typeof (x as PlannedScene).description === "string" &&
                        typeof (x as PlannedScene).script === "string" &&
                        typeof (x as PlannedScene).clipsNeeded === "number",
                )
            ) {
                return parsed as PlannedScene[];
            }
        } catch (err) {
            console.warn(
                `[duration-aware-video] failed to read existing scenes.json:`,
                err instanceof Error ? err.message : err,
            );
        }
        return null;
    }

    /**
     * Duration-aware document-to-video orchestrator.
     *
     * content → scenes(description+script) → planned scenes (with audio duration & clipsNeeded)
     *   → shared style brief → N keyframe images per scene
     *   → N Veo clips per scene with last-frame chaining
     *   → per-scene concat at full Veo clip length (no hard trim to narration estimate)
     *   → final concat of all stitched scenes.
     *
     * Mirrors the structure / logging / retry / partial-recovery semantics of
     * {@link generateSceneImageMontageFromContent}.
     */
    async generateDurationAwareVideoFromContent(
        content: string,
        aspectRatio: "9:16" | "16:9" = "9:16",
        resume: SceneImageMontageResumeOptions = {},
    ) {
        console.log("[duration-aware-video] Pipeline start: scene-with-narration duration-aware montage.");
        // Same preflight as the other pipeline — fail fast on missing ffmpeg before spending Veo cost.
        await this.assertFfmpegAvailable();
        console.log("[duration-aware-video] Preflight complete: ffmpeg is available.");

        const outDir = path.resolve(process.cwd(), "generated-document-to-video");
        await fs.mkdir(outDir, { recursive: true });
        console.log("[duration-aware-video] Step complete: output directory ready.", outDir);

        const resumeMontageId = this.normalizeResumeMontageId(resume.resumeMontageId);
        const resumeFromScene = this.normalizeResumeFromScene(resume.resumeFromScene);
        const montageId = resumeMontageId ?? `montage_dur_${Date.now()}`;
        const montageDir = path.join(outDir, montageId);
        const imagesDir = path.join(montageDir, "images");
        const clipsDir = path.join(montageDir, "clips");
        const scenesDir = path.join(montageDir, "scenes");
        await fs.mkdir(imagesDir, { recursive: true });
        await fs.mkdir(clipsDir, { recursive: true });
        await fs.mkdir(scenesDir, { recursive: true });
        console.log("[duration-aware-video] Step complete: montage workspace ready.", {
            montageDir,
            imagesDir,
            clipsDir,
            scenesDir,
            resumeFromScene,
            isResume: !!resumeMontageId,
        });
        // Surface a copy-pasteable resume hint NOW (before the long-running phases). If the
        // server later dies mid-run, the user can simply re-POST with the same montageId and
        // the pipeline will skip every keyframe/clip/scene that already landed on disk.
        console.log(
            `[duration-aware-video] Resume hint: if this run is interrupted, re-POST /gemini/document-to-video?resumeMontageId=${montageId} (optionally &resumeFromScene=<N>) with the same document to continue from the last completed step.`,
        );

        let failedStep = "initialization";
        let scenes: PlannedScene[] = [];
        let styleBrief = "";
        const allImagePaths: string[] = [];
        const allClipFiles: string[] = [];
        const stitchedScenes: StitchedScene[] = [];
        let anchorBase64: string | undefined;
        let anchorMime = "image/png";

        try {
            // Step 1: scene plan (reuse existing plan when resuming so clip counts stay stable).
            failedStep = "scene generation";
            if (resumeMontageId) {
                const reused = await this.loadPlannedScenesFromDisk(montageDir);
                if (reused?.length) {
                    scenes = reused;
                    console.log(
                        `[duration-aware-video] Resume: loaded existing scenes.json (count=${scenes.length}).`,
                    );
                }
            }
            if (!scenes.length) {
                scenes = await this.planDurationAwareScenesFromContent(content);
                await fs.writeFile(
                    path.join(montageDir, "scenes.json"),
                    JSON.stringify(scenes, null, 2),
                    "utf8",
                );
            }
            console.log("[duration-aware-video] the scenes are ", scenes);
            if (resumeFromScene > scenes.length) {
                throw new Error(
                    `resumeFromScene (${resumeFromScene}) is greater than planned scene count (${scenes.length}).`,
                );
            }
            const totalClips = scenes.reduce((sum, s) => sum + s.clipsNeeded, 0);
            console.log(
                `[duration-aware-video] Step complete: scenes planned (count=${scenes.length}, totalClips=${totalClips}).`,
            );

            // Step 2: shared visual style bible across all scenes (same call signature as the other pipeline).
            failedStep = "style brief generation";
            styleBrief = await this.runStepWithRetries("style brief generation", () =>
                this.geminiVisualStyleBrief(
                    content,
                    scenes.map((s) => s.description),
                ),
            );
            console.log(
                `[duration-aware-video] Step complete: style brief ready (length=${styleBrief.length} chars).`,
            );

            // Try reusing the very first frame of the very first scene as the visual anchor.
            const earlyAnchor = await this.resolveExistingFramePath(imagesDir, 1, 1);
            if (earlyAnchor) {
                const buf = await fs.readFile(earlyAnchor);
                anchorBase64 = buf.toString("base64");
                anchorMime = earlyAnchor.toLowerCase().endsWith(".jpg") || earlyAnchor.toLowerCase().endsWith(".jpeg")
                    ? "image/jpeg"
                    : "image/png";
                console.log("[duration-aware-video] Reusing first scene/frame as visual anchor.");
            }

            // Step 3: per-scene keyframe images (start / mid-i / end based on clipsNeeded).
            console.log(
                `[duration-aware-video] Phase: generating ${totalClips} keyframe images across ${scenes.length} scenes...`,
            );
            for (let i = 0; i < scenes.length; i++) {
                const scene = scenes[i];
                const labels = this.frameLabelsForClipCount(scene.clipsNeeded);
                const referenceImages: string[] = [];

                for (let f = 0; f < scene.clipsNeeded; f++) {
                    const frameLabel = labels[f];
                    const stepLabel = `keyframe scene ${i + 1}/${scenes.length} frame ${f + 1}/${scene.clipsNeeded} (${frameLabel})`;

                    const existing = await this.resolveExistingFramePath(imagesDir, i + 1, f + 1);
                    if (existing) {
                        referenceImages.push(existing);
                        allImagePaths.push(existing);
                        if (!anchorBase64) {
                            const buf = await fs.readFile(existing);
                            anchorBase64 = buf.toString("base64");
                            anchorMime = existing.toLowerCase().endsWith(".jpg") || existing.toLowerCase().endsWith(".jpeg")
                                ? "image/jpeg"
                                : "image/png";
                        }
                        console.log(
                            `[duration-aware-video] ${stepLabel}: existing image found, skipping generation -> ${existing}`,
                        );
                        continue;
                    }

                    // Skip image generation for scenes earlier than the resume point — caller is
                    // expected to have those on disk already from a previous run.
                    if (i + 1 < resumeFromScene) {
                        throw new Error(
                            `Resume requested from scene ${resumeFromScene} but missing keyframe for scene ${i + 1} frame ${f + 1}.`,
                        );
                    }

                    failedStep = stepLabel;
                    const augmentedPrompt = scene.clipsNeeded > 1
                        ? `${scene.description}\n\nFrame stage: ${frameLabel} (frame ${f + 1} of ${scene.clipsNeeded} for this scene). Keep identical subjects, wardrobe, props, environment, and lighting between frames; only vary composition or motion stage to convey the ${frameLabel} of one continuous shot.`
                        : scene.description;

                    const packed = await this.runStepWithRetries(stepLabel, async () => {
                        try {
                            return await this.generateSceneKeyframeFlashImage({
                                sceneIndex: i,
                                total: scenes.length,
                                scenePrompt: augmentedPrompt,
                                styleBrief,
                                referenceBase64: anchorBase64,
                                referenceMime: anchorMime,
                                aspectRatio,
                            });
                        } catch (err: any) {
                            console.warn(
                                `[duration-aware-video] flash-image failed (${stepLabel}); falling back to Imagen:`,
                                err?.message ?? err,
                            );
                            const combined = `STYLE BIBLE (keep consistent):\n${styleBrief}\n\nScene ${i + 1} of ${scenes.length}, ${frameLabel} frame — cinematic educational keyframe, no text overlays:\n${augmentedPrompt}`;
                            return await this.generateSceneKeyframeImagen(combined, aspectRatio, {
                                index1Based: i + 1,
                                total: scenes.length,
                            });
                        }
                    });

                    if (!anchorBase64) {
                        anchorBase64 = packed.base64;
                        anchorMime = packed.mimeType;
                        console.log(
                            "[duration-aware-video] Step complete: visual anchor set from first keyframe.",
                        );
                    }

                    const ext =
                        packed.mimeType.includes("jpeg") || packed.mimeType.includes("jpg") ? "jpg" : "png";
                    const imagePath = path.join(
                        imagesDir,
                        `scene_${String(i + 1).padStart(2, "0")}_frame_${String(f + 1).padStart(2, "0")}.${ext}`,
                    );
                    await fs.writeFile(imagePath, Buffer.from(packed.base64, "base64"));
                    referenceImages.push(imagePath);
                    allImagePaths.push(imagePath);
                    console.log(
                        `[duration-aware-video] Step complete: ${stepLabel} saved to disk -> ${imagePath}`,
                    );
                }

                scene.referenceImages = referenceImages;
            }
            console.log("[duration-aware-video] Phase complete: all keyframe images written.");

            // Step 4 + 5: per-scene Veo clip chain → per-scene stitch + trim.
            console.log(
                `[duration-aware-video] Phase: generating Veo clips and stitching per-scene videos...`,
            );
            for (let i = 0; i < scenes.length; i++) {
                const scene = scenes[i];
                const sceneTag = `scene_${String(i + 1).padStart(2, "0")}`;
                const sceneClipDir = path.join(clipsDir, sceneTag);
                await fs.mkdir(sceneClipDir, { recursive: true });

                const stitchedScenePath = path.join(scenesDir, `${sceneTag}.mp4`);
                const sceneClipFiles: string[] = [];
                let previousLastFramePath: string | undefined;

                const sanitizeForQuotedDialogue = (s: string) =>
                    s
                        .replace(/\s+/g, " ")
                        .replace(/["“”]/g, "'")
                        .trim();

                const splitNarrationIntoClipSegments = (script: string, clipCount: number): string[] => {
                    const clean = sanitizeForQuotedDialogue(script);
                    if (!clean) return Array.from({ length: clipCount }, () => "");
                    if (clipCount <= 1) return [clean];

                    const words = clean.split(" ").filter(Boolean);
                    if (words.length <= clipCount) {
                        // Not enough words to split meaningfully; keep it in the first clip.
                        return [clean, ...Array.from({ length: clipCount - 1 }, () => "")];
                    }

                    const segments: string[] = [];
                    for (let idx = 0; idx < clipCount; idx++) {
                        const start = Math.floor((idx * words.length) / clipCount);
                        const end = Math.floor(((idx + 1) * words.length) / clipCount);
                        const part = words.slice(start, end).join(" ").trim();
                        segments.push(part);
                    }
                    return segments;
                };

                const narrationSegments = splitNarrationIntoClipSegments(scene.script ?? "", scene.clipsNeeded);

                for (let c = 0; c < scene.clipsNeeded; c++) {
                    const clipPath = path.join(
                        sceneClipDir,
                        `clip_${String(c + 1).padStart(2, "0")}.mp4`,
                    );
                    const lastFramePath = clipPath.replace(/\.mp4$/i, "_lastframe.jpg");
                    const clipLabel = `scene ${i + 1}/${scenes.length} clip ${c + 1}/${scene.clipsNeeded}`;

                    const existingClipBytes = await this.clipFileByteSize(clipPath);
                    if (existingClipBytes > 0) {
                        console.log(
                            `[duration-aware-video] ${clipLabel}: existing clip found, skipping Veo generation -> ${clipPath}`,
                        );
                        sceneClipFiles.push(clipPath);
                        allClipFiles.push(clipPath);

                        // Need the last frame on disk for chaining the next clip even on resume.
                        if (c < scene.clipsNeeded - 1) {
                            if (!(await this.fileExists(lastFramePath))) {
                                await this.extractLastFrame(clipPath, lastFramePath);
                            }
                            previousLastFramePath = lastFramePath;
                        }
                        continue;
                    }

                    if (i + 1 < resumeFromScene) {
                        throw new Error(
                            `Resume requested from scene ${resumeFromScene} but missing clip ${c + 1} for scene ${i + 1}.`,
                        );
                    }

                    // Image input priority per APPROACH.md: previous clip's last frame (if any),
                    // else the per-clip reference image, else fall back to the first keyframe.
                    const startImagePath =
                        previousLastFramePath ?? scene.referenceImages[c] ?? scene.referenceImages[0];
                    if (!startImagePath || !(await this.fileExists(startImagePath))) {
                        throw new Error(
                            `Missing start image for ${clipLabel}. Expected a keyframe or chained last frame on disk.`,
                        );
                    }
                    const imageBuf = await fs.readFile(startImagePath);
                    const imageBytes = imageBuf.toString("base64");
                    const mimeType =
                        startImagePath.toLowerCase().endsWith(".jpg") ||
                            startImagePath.toLowerCase().endsWith(".jpeg")
                            ? "image/jpeg"
                            : "image/png";

                    const continuityHint = c === 0
                        ? "Open this scene with cinematic motion that establishes the situation."
                        : `This is clip ${c + 1} of ${scene.clipsNeeded} for the same continuous scene. The provided image is the FINAL frame of the previous clip — continue the same shot without cutting, jumping, or changing subjects, wardrobe, props, environment, or camera angle. Pick up motion exactly where it left off.`;

                    const narration = narrationSegments[c] ?? "";
                    const audioHint = narration
                        ? `\n\nAudio / voice-over cues:\n- Narrator (calm instructional voice-over, neutral accent): "${narration}"`
                        : `\n\nAudio / voice-over cues:\n- No dialogue. Ambient room tone only.`;

                    const videoPrompt =
                        `Scene ${i + 1} of ${scenes.length}, clip ${c + 1} of ${scene.clipsNeeded}.\n` +
                        `${continuityHint}\n` +
                        `Maintain consistent characters, wardrobe, props, environment, and lighting throughout.\n\n` +
                        `Visual:\n${scene.description}` +
                        audioHint;

                    failedStep = `video clip ${clipLabel}`;
                    console.log(`[duration-aware-video] Starting Veo ${clipLabel}...`);
                    await this.runMontageVeoClipWithRetries({
                        clipIndex1Based: c + 1,
                        aspectRatio,
                        imageBytes,
                        mimeType,
                        videoPrompt,
                        downloadPath: clipPath,
                        label: clipLabel,
                    });
                    console.log(
                        `[duration-aware-video] Step complete: ${clipLabel} downloaded -> ${clipPath}`,
                    );

                    sceneClipFiles.push(clipPath);
                    allClipFiles.push(clipPath);

                    if (c < scene.clipsNeeded - 1) {
                        await this.extractLastFrame(clipPath, lastFramePath);
                        previousLastFramePath = lastFramePath;
                        console.log(
                            `[duration-aware-video] Step complete: extracted last frame for chaining -> ${lastFramePath}`,
                        );
                    }
                }

                // Stitch per-scene video (full concat length; no hard trim to narration estimate).
                failedStep = `stitch scene ${i + 1}/${scenes.length}`;
                const existingStitchedBytes = await this.clipFileByteSize(stitchedScenePath);
                if (existingStitchedBytes > 0) {
                    console.log(
                        `[duration-aware-video] scene ${i + 1}/${scenes.length}: existing stitched video found, skipping stitch -> ${stitchedScenePath}`,
                    );
                } else {
                    console.log(
                        `[duration-aware-video] Stitching scene ${i + 1}/${scenes.length}: ${sceneClipFiles.length} clip(s) (no narration-duration trim).`,
                    );
                    await this.stitchSceneClips(sceneClipFiles, stitchedScenePath);
                    console.log(
                        `[duration-aware-video] Step complete: scene ${i + 1}/${scenes.length} stitched -> ${stitchedScenePath}`,
                    );
                }

                const probedDuration = await this.probeVideoDurationSeconds(stitchedScenePath);
                const sceneVideoDurationSeconds = probedDuration ?? scene.audioDurationSeconds;
                if (probedDuration == null) {
                    console.warn(
                        `[duration-aware-video] scene ${i + 1}/${scenes.length}: ffprobe duration unavailable; using narration estimate ${scene.audioDurationSeconds}s for totals.`,
                    );
                }

                stitchedScenes.push({
                    sceneId: scene.id,
                    sceneIndex: scene.sceneIndex,
                    videoPath: stitchedScenePath,
                    trimmedDurationSeconds: sceneVideoDurationSeconds,
                });
            }
            console.log("[duration-aware-video] Phase complete: all per-scene videos stitched.");

            // Step 6: final concat of all per-scene videos.
            failedStep = "final stitching";
            const combinedFileName = `${montageId}.mp4`;
            const combinedFilePath = path.join(outDir, combinedFileName);
            const stitchedPaths = stitchedScenes.map((s) => s.videoPath);

            console.log("[duration-aware-video] Phase: stitching all scene videos with FFmpeg...");
            try {
                await this.concatClipsFFmpeg(stitchedPaths, combinedFilePath);
                console.log(
                    "[duration-aware-video] Step complete: FFmpeg concat (stream copy) succeeded.",
                );
            } catch {
                console.log(
                    "[duration-aware-video] Stream copy concat failed; falling back to re-encode concat.",
                );
                await this.concatClipsFFmpegReencode(stitchedPaths, combinedFilePath);
                console.log(
                    "[duration-aware-video] Step complete: FFmpeg re-encode concat succeeded.",
                );
            }

            const totalDurationSeconds = stitchedScenes.reduce(
                (sum, s) => sum + s.trimmedDurationSeconds,
                0,
            );
            console.log("[duration-aware-video] Pipeline complete: final montage ->", combinedFilePath);

            return {
                combinedFilePath,
                combinedFileName,
                montageId,
                montageDir,
                imagesDir,
                clipsDir,
                scenesDir,
                scenes,
                scenesStitched: stitchedScenes,
                clipFiles: allClipFiles,
                imagePaths: allImagePaths,
                styleBrief,
                totalDurationSeconds,
            };
        } catch (err: any) {
            const partial = await this.stitchSuccessfulScenes({
                stitchedScenes,
                outDir,
                montageId,
            });
            const originalErrorMessage =
                err instanceof Error ? err.message : typeof err === "string" ? err : JSON.stringify(err);
            const details: SceneImageMontageFailureDetails = {
                step: failedStep,
                montageId,
                montageDir,
                imagesDir,
                retriesPerStep: PIPELINE_STEP_MAX_RETRIES,
                successfulImageCount: allImagePaths.length,
                successfulClipCount: partial.stitchedSceneCount,
                partialCombinedFilePath: partial.partialCombinedFilePath,
                partialCombinedFileName: partial.partialCombinedFileName,
                originalErrorMessage,
            };
            throw new SceneImageMontageError(
                `[duration-aware-video] failed at ${failedStep} after retries: ${originalErrorMessage}`,
                details,
            );
        }
    }
}