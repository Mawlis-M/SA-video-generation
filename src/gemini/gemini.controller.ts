import { Body, Controller, Post } from "@nestjs/common";
import { GeminiService } from "./gemini.service";
import { GenerateVideoDto, GenerateVideoMontageDto } from "./dto/generate-video.dto";
import * as path from "path";
import * as fs from "fs/promises";

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
}