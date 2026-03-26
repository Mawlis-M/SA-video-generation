import { ArrayMinSize, IsArray, IsIn, IsOptional, IsString, MinLength } from "class-validator";

export class GenerateVideoDto {
  @IsString()
  @MinLength(5)
  prompt: string;

  @IsOptional()
  @IsIn(["9:16", "16:9"])
  aspectRatio?: "9:16" | "16:9";
}

export class GenerateVideoMontageDto {
    @IsArray()
    @ArrayMinSize(1)
    @IsString({ each: true })
    @MinLength(5, { each: true })
    scenes: string[];
  
    @IsOptional()
    @IsIn(["9:16", "16:9"])
    aspectRatio?: "9:16" | "16:9";
  }

/** Same body shape as a text-only scene montage, but runs image keyframes then image-to-video. */
export class SceneImageMontageDto {
  @IsString()
  @MinLength(5)
  content: string;

  @IsOptional()
  @IsIn(["9:16", "16:9"])
  aspectRatio?: "9:16" | "16:9";
}