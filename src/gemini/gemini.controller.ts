import { Body, Controller, Post } from "@nestjs/common";
import { GeminiService } from "./gemini.service";
import { GenerateVideoMontageDto, SceneImageMontageDto } from "./dto/generate-video.dto";

@Controller("gemini")
export class GeminiController {
  constructor(private readonly gemini: GeminiService) { }

  @Post("generate")
  async generate(@Body() dto: GenerateVideoMontageDto) {
    const result = await this.gemini.generateSingleClipToFile(dto.scenes, dto.aspectRatio ?? "9:16");
    return result; // { filePath, fileName }
  }


  @Post("montage")
  async montage(@Body() reqparams: any) {
    const result = await this.gemini.generateMontageToFile(reqparams.scenes, reqparams.aspectRatio ?? "9:16");
    return result; // { filePath, fileName }
  }

  @Post("scene")
  async scene(@Body() reqparams: any) {
    const scenes = await this.gemini.geminiVideoScript(reqparams.content);
    console.log("scenes", scenes);
    const result = await this.gemini.generateMontageToFile(scenes, reqparams.aspectRatio ?? "9:16");
    return result; // { filePath, fileName }
  }

  /**
   * content → scenes → shared style brief → per-scene keyframe images (under generated/.../images/)
   * → Veo image-to-video per scene → FFmpeg concat (same output shape as /scene plus image paths).
   */
  @Post("scene-with-images")
  async sceneWithImages(@Body() dto: SceneImageMontageDto) {
    return await this.gemini.generateSceneImageMontageFromContent(
      dto.content,
      dto.aspectRatio ?? "9:16",
    );
  }
}