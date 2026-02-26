import { Injectable } from "@nestjs/common";
import { ConfigService } from "@nestjs/config";
import { GoogleGenAI } from "@google/genai";
import * as path from "path";
import * as fs from "fs/promises";
import { spawn } from "child_process";

function sleep(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

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