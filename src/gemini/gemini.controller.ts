import { Body, Controller, HttpException, Post } from "@nestjs/common";
import { GeminiService } from "./gemini.service";
import { GenerateScenesAndExtendToTwoMinutesDto, GenerateVideoMontageDto } from "./dto/generate-video.dto";

/** Maps unknown errors to HTTP responses with a readable `message` (avoids opaque 500s). */
function handleGeminiRouteError(err: unknown, operation: string): never {
  if (err instanceof HttpException) throw err;

  let message: string;
  if (err instanceof Error) {
    message = err.message;
  } else if (typeof err === "string") {
    message = err;
  } else {
    try {
      message = JSON.stringify(err);
    } catch {
      message = String(err);
    }
  }

  console.error(`[GeminiController] ${operation}`, err);

  throw new HttpException(
    {
      statusCode: 502,
      message,
      error: "Bad Gateway",
      operation,
    },
    502,
  );
}

@Controller("gemini")
export class GeminiController {
  constructor(private readonly gemini: GeminiService) {}

  @Post("generate")
  async generate(@Body() dto: GenerateVideoMontageDto) {
    try {
      return await this.gemini.generateSingleClipToFile(dto.scenes, dto.aspectRatio ?? "9:16");
    } catch (err) {
      handleGeminiRouteError(err, "POST /gemini/generate");
    }
  }

  @Post("montage")
  async montage(@Body() reqparams: any) {
    try {
      return await this.gemini.generateMontageToFile(reqparams.scenes, reqparams.aspectRatio ?? "9:16");
    } catch (err) {
      handleGeminiRouteError(err, "POST /gemini/montage");
    }
  }

  @Post("scene")
  async scene(@Body() reqparams: any) {
    try {
      const scenes = await this.gemini.geminiVideoScript(reqparams.content);
      console.log("scenes", scenes);
      return await this.gemini.generateMontageToFile(scenes, reqparams.aspectRatio ?? "9:16");
    } catch (err) {
      handleGeminiRouteError(err, "POST /gemini/scene");
    }
  }

  /**
   * POST /gemini/extend
   * Same pipeline as `/gemini/scenes-and-extend`: `content` → scenes → Veo create + chained extends;
   * final file is `generated/{completionTimestamp}.mp4`.
   */
  @Post("extend")
  async extend(@Body() dto: GenerateScenesAndExtendToTwoMinutesDto) {
    try {
      return await this.gemini.generateScenesAndExtendToTwoMinutes(dto.content, dto.aspectRatio ?? "9:16");
    } catch (err) {
      handleGeminiRouteError(err, "POST /gemini/extend");
    }
  }

  /**
   * POST /gemini/scenes-and-extend
   * Same as `/gemini/extend`: ~2 min from instructional `content`, saved as `{completionTimestamp}.mp4`.
   */
  @Post("scenes-and-extend")
  async scenesAndExtend(@Body() dto: GenerateScenesAndExtendToTwoMinutesDto) {
    try {
      return await this.gemini.generateScenesAndExtendToTwoMinutes(dto.content, dto.aspectRatio ?? "9:16");
    } catch (err) {
      handleGeminiRouteError(err, "POST /gemini/scenes-and-extend");
    }
  }
}