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
    }): Promise<void> {
        const label = `clip ${args.clipIndex1Based}`;
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
}