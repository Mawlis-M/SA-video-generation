import {
  BadRequestException,
  Controller,
  Post,
  UploadedFile,
  UseInterceptors,
} from "@nestjs/common";
import { FileInterceptor } from "@nestjs/platform-express";
import type { Express } from "express";
import { DocumentTopicsService } from "./document-topics.service";

const MAX_FILE_BYTES = 15 * 1024 * 1024;

@Controller("documents")
export class DocumentTopicsController {
  constructor(private readonly documentTopics: DocumentTopicsService) {}

  /**
   * Upload a document (PDF, DOCX, TXT, MD, etc.), extract text, and return
   * sections grouped by topic in original document order.
   */
  @Post("analyze-topics")
  @UseInterceptors(
    FileInterceptor("file", {
      limits: { fileSize: MAX_FILE_BYTES },
    }),
  )
  async analyzeTopics(@UploadedFile() file: Express.Multer.File | undefined) {
    console.log("[DocumentTopics] /documents/analyze-topics request received");
    if (!file?.buffer?.length) {
      console.log("[DocumentTopics] request rejected: missing or empty file");
      throw new BadRequestException(
        'Missing file: send multipart form field "file" with a document.',
      );
    }
    console.log("[DocumentTopics] file accepted", {
      originalName: file.originalname,
      mimeType: file.mimetype,
      sizeBytes: file.size,
    });

    console.log("[DocumentTopics] starting analysis pipeline");
    return await this.documentTopics.analyzeUploadedDocument(
      file.buffer,
      file.mimetype,
      file.originalname,
    );
  }
}
