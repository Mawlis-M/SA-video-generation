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
   * Upload a document → extract topics → generate scene breakdowns per topic (via Gemini, no video generation)
   * → return estimated clip count, scene list, and approximate total video duration.
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
      const analysis = await this.documentTopics.analyzeTopics(file);
      const ar: "9:16" | "16:9" =
        aspectRatio === "16:9" ? "16:9" : "9:16";

      const topicContents = analysis.topics.map((topic) => {
        const title = (topic.title ?? "").trim();
        const body = (topic.content ?? "").trim();
        return title ? `${title}\n\n${body}` : body;
      });

      const VEO_CLIP_DURATION_SECONDS = 8;

      const topicBreakdowns: Array<{
        topicIndex: number;
        title: string;
        sceneCount: number;
        scenes: string[];
        estimatedDurationSeconds: number;
      }> = [];

      let totalScenes = 0;

      for (let i = 0; i < topicContents.length; i++) {
        const content = topicContents[i];
        if (!content.trim()) continue;

        console.log(`[estimate-duration] Generating scene breakdown for topic ${i + 1}/${topicContents.length}...`);
        const scenes = await this.gemini.geminiVideoScript(content);
        console.log("scenes presently here is ", scenes);
        console.log(`[estimate-duration] Topic ${i + 1}: ${scenes.length} scenes.`);

        totalScenes += scenes.length;
        topicBreakdowns.push({
          topicIndex: i,
          title: analysis.topics[i]?.title ?? "",
          sceneCount: scenes.length,
          scenes,
          estimatedDurationSeconds: scenes.length * VEO_CLIP_DURATION_SECONDS,
        });
      }

      const totalDurationSeconds = totalScenes * VEO_CLIP_DURATION_SECONDS;

      return {
        originalFileName: analysis.originalFileName,
        extractedTextLength: analysis.extractedTextLength,
        topicCount: analysis.topics.length,
        aspectRatio: ar,
        totalScenes,
        clipDurationSeconds: VEO_CLIP_DURATION_SECONDS,
        totalEstimatedDurationSeconds: totalDurationSeconds,
        totalEstimatedDurationFormatted: `${Math.floor(totalDurationSeconds / 60)}m ${totalDurationSeconds % 60}s`,
        topicBreakdowns,
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
   * Upload a document → extract topics → one POST /gemini/scene-with-images pipeline per topic (sequential).
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
      const analysis = await this.documentTopics.analyzeTopics(file);
      const ar: "9:16" | "16:9" =
        aspectRatio === "16:9" ? "16:9" : "9:16";

      const topicContents = analysis.topics.map((topic) => {
        const title = (topic.title ?? "").trim();
        const body = (topic.content ?? "").trim();
        return title ? `${title}\n\n${body}` : body;
      });

      type MontageResult = Awaited<
        ReturnType<GeminiService["generateSceneImageMontageFromContent"]>
      >;
      const montagesByTopic: Array<
        { topicIndex: number; title: string } & MontageResult
      > = [];

      for (let i = 0; i < topicContents.length; i++) {
        const content = topicContents[i];
        console.log("content presently here is ", content);
        if (!content.trim()) {
          continue;
        }
        console.log(`[document-to-video] Starting montage for topic ${i + 1}...`);
        try {
          const montage = await this.gemini.generateSceneImageMontageFromContent(
            content,
            ar,
          );
          console.log(`[document-to-video] Topic ${i + 1} montage complete.`);
          montagesByTopic.push({
            topicIndex: i,
            title: analysis.topics[i]?.title ?? "",
            ...montage,
          });
        } catch (err: any) {
          if (err instanceof SceneImageMontageError) {
            throw new HttpException(
              {
                error: "Document to video failed after retries for one topic.",
                message: err.message,
                failedTopicIndex: i,
                failedTopicTitle: analysis.topics[i]?.title ?? "",
                completedTopics: montagesByTopic,
                failureDetails: err.details,
              },
              HttpStatus.BAD_GATEWAY,
            );
          }
          throw err;
        }
      }

      return {
        originalFileName: analysis.originalFileName,
        extractedTextLength: analysis.extractedTextLength,
        topicCount: analysis.topics.length,
        topics: analysis.topics,
        topicContents,
        aspectRatio: ar,
        montagesByTopic,
      };
    } catch (err: any) {
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
