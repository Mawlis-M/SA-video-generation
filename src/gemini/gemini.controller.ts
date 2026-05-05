import {
  Body,
  Controller,
  HttpException,
  HttpStatus,
  Post,
  Query,
  UploadedFile,
  UseInterceptors,
} from "@nestjs/common";
import { GeminiService, SceneImageMontageError } from "./gemini.service";
import { GenerateVideoMontageDto, SceneImageMontageDto } from "./dto/generate-video.dto";
import { DocumentTopicsService } from "src/document-topics/document-topics.service";
import { FileInterceptor } from "@nestjs/platform-express";

import * as path from "path";
import * as fs from "fs/promises";

const MAX_FILE_BYTES = 15 * 1024 * 1024;
@Controller("gemini")
export class GeminiController {
  constructor(
    private readonly gemini: GeminiService,
    private readonly documentTopics: DocumentTopicsService,
  ) { }

  @Post("generate")
  async generate(@Body() dto: GenerateVideoMontageDto) {
    const result = await this.gemini.generateSingleClipToFile(
      dto.scenes,
      dto.aspectRatio ?? "9:16",
    );
    return result; // { filePath, fileName }
  }

  @Post("montage")
  async montage(@Body() reqparams: any) {
    const result = await this.gemini.generateMontageToFile(
      reqparams.scenes,
      reqparams.aspectRatio ?? "9:16",
    );
    return result; // { filePath, fileName }
  }

  @Post("stitch")
  async stitch(@Body() reqparams: any) {
    const clipFiles: string[] = Array.isArray(reqparams)
      ? reqparams
      : reqparams?.clipFiles;

    if (!Array.isArray(clipFiles) || clipFiles.length === 0) {
      throw new HttpException(
        { error: "Invalid request body. Provide clipFiles as a non-empty string array." },
        HttpStatus.BAD_REQUEST,
      );
    }

    const invalidClip = clipFiles.find((clip) => typeof clip !== "string" || !clip.trim());
    if (invalidClip !== undefined) {
      throw new HttpException(
        { error: "Each clip file path must be a non-empty string." },
        HttpStatus.BAD_REQUEST,
      );
    }

    const outDir = path.resolve(process.cwd(), "generated");
    await fs.mkdir(outDir, { recursive: true });

    const montageId = `montage_${Date.now()}`;
    const combinedFileName = `${montageId}.mp4`;
    const combinedFilePath = path.join(outDir, combinedFileName);
    try {
      await this.gemini.concatClipsFFmpeg(clipFiles, combinedFilePath);
    } catch (err) {
      await this.gemini.concatClipsFFmpegReencode(clipFiles, combinedFilePath);
    }

    return {
      combinedFilePath,
      combinedFileName,
      clipFiles,
    };
  }

  @Post("scene")
  async scene(@Body() reqparams: any) {
    const scenes = await this.gemini.geminiVideoScript(reqparams.content);
    console.log("scenes", scenes);
    const result = await this.gemini.generateMontageToFile(
      scenes,
      reqparams.aspectRatio ?? "9:16",
    );
    return result; // { filePath, fileName }
  }

  /**
   * content → scenes → shared style brief → per-scene keyframe images (under generated/.../images/)
   * → Veo image-to-video per scene → FFmpeg concat (same output shape as /scene plus image paths).
   */
  @Post("scene-with-images")
  async sceneWithImages(@Body() dto: SceneImageMontageDto) {
    try {
      return await this.gemini.generateSceneImageMontageFromContent(
        dto.content,
        dto.aspectRatio ?? "9:16",
      );
    } catch (err: any) {
      console.error("[controller scene-with-images] Unhandled error:", err?.message, err?.stack, err);
      throw new HttpException(
        { error: err?.message ?? "Unknown error in scene-with-images pipeline" },
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  /**
   * Upload a document → extract full text → summarize whole document (OpenAI)
   * → generate scene breakdown from summary (via Gemini, no video generation)
   * → return estimated clip count and approximate total video duration.
   */
  @Post("estimate-duration")
  @UseInterceptors(
    FileInterceptor("file", {
      limits: { fileSize: MAX_FILE_BYTES },
    }),
  )
  async estimateDuration(
    @UploadedFile() file: Express.Multer.File,
    @Query("aspectRatio") aspectRatio?: "9:16" | "16:9",
  ) {
    try {
      if (!file?.buffer?.length) {
        throw new HttpException(
          { error: 'Missing file: send multipart form field "file" with a document.' },
          HttpStatus.BAD_REQUEST,
        );
      }

      const analysis = await this.documentTopics.summarizeUploadedDocument(
        file.buffer,
        file.mimetype,
        file.originalname,
      );
      const ar: "9:16" | "16:9" =
        aspectRatio === "16:9" ? "16:9" : "9:16";

      const VEO_CLIP_DURATION_SECONDS = 8;
      console.log(
        `[estimate-duration] Summary generated (chars=${analysis.summaryTextLength}). Generating scenes...`,
      );
      const scenes = await this.gemini.geminiVideoScript(analysis.summary);
      const totalScenes = scenes.length;

      const totalDurationSeconds = totalScenes * VEO_CLIP_DURATION_SECONDS;

      return {
        originalFileName: analysis.originalFileName,
        extractedTextLength: analysis.extractedTextLength,
        summaryTextLength: analysis.summaryTextLength,
        summary: analysis.summary,
        aspectRatio: ar,
        totalScenes,
        clipDurationSeconds: VEO_CLIP_DURATION_SECONDS,
        totalEstimatedDurationSeconds: totalDurationSeconds,
        totalEstimatedDurationFormatted: `${Math.floor(totalDurationSeconds / 60)}m ${totalDurationSeconds % 60}s`,
        scenes,
      };
    } catch (err: any) {
      console.error("[controller estimate-duration] Unhandled error:", err?.message, err?.stack, err);
      throw new HttpException(
        { error: err?.message ?? "Unknown error in estimate-duration pipeline" },
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  /**
   * Upload a document → extract full text → summarize whole document (OpenAI)
   * → one POST /gemini/scene-with-images pipeline from summary.
   * Optional query: ?aspectRatio=9:16 | 16:9 (default 9:16).
   */
  @Post("document-to-video")
  @UseInterceptors(
    FileInterceptor("file", {
      limits: { fileSize: MAX_FILE_BYTES },
    }),
  )
  async documentToVideo(
    @UploadedFile() file: Express.Multer.File,
    @Query("aspectRatio") aspectRatio?: "9:16" | "16:9",
  ) {
    try {
      if (!file?.buffer?.length) {
        throw new HttpException(
          { error: 'Missing file: send multipart form field "file" with a document.' },
          HttpStatus.BAD_REQUEST,
        );
      }

      const analysis = await this.documentTopics.summarizeUploadedDocument(
        file.buffer,
        file.mimetype,
        file.originalname,
      );
      const ar: "9:16" | "16:9" =
        aspectRatio === "16:9" ? "16:9" : "9:16";
      console.log(
        `[document-to-video] Summary generated (chars=${analysis.summaryTextLength}). Starting single montage...`,
      );
      const montage = await this.gemini.generateSceneImageMontageFromContent(
        analysis.summary,
        ar,
      );

      return {
        originalFileName: analysis.originalFileName,
        extractedTextLength: analysis.extractedTextLength,
        summaryTextLength: analysis.summaryTextLength,
        summary: analysis.summary,
        aspectRatio: ar,
        montage,
      };
    } catch (err: any) {
      if (err instanceof SceneImageMontageError) {
        throw new HttpException(
          {
            error: "Document to video failed while generating montage from summary.",
            message: err.message,
            failureDetails: err.details,
          },
          HttpStatus.BAD_GATEWAY,
        );
      }
      if (err instanceof HttpException) {
        throw err;
      }
      console.error("[controller document-to-video] Unhandled error:", err?.message, err?.stack, err);
      throw new HttpException(
        { error: err?.message ?? "Unknown error in document-to-video pipeline" },
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }
}
