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

export class GenerateScenesAndExtendToTwoMinutesDto {
  @IsString()
  @MinLength(5)
  content: string;

  @IsOptional()
  @IsIn(["9:16", "16:9"])
  aspectRatio?: "9:16" | "16:9";
}