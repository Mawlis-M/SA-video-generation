import { Module } from "@nestjs/common";
import { GeminiService } from "./gemini.service";
import { GeminiController } from "./gemini.controller";
import { DocumentTopicsModule } from "src/document-topics/document-topics.module";
@Module({
  imports: [DocumentTopicsModule],
  providers: [GeminiService],
  controllers: [GeminiController],
})
export class VeoModule {}