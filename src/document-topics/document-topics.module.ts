import { Module } from "@nestjs/common";
import { DocumentTopicsController } from "./document-topics.controller";
import { DocumentTopicsService } from "./document-topics.service";

@Module({
  controllers: [DocumentTopicsController],
  providers: [DocumentTopicsService],
  exports: [DocumentTopicsService],
})
export class DocumentTopicsModule {}
