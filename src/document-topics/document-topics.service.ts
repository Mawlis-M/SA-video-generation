import {
  BadRequestException,
  Injectable,
  InternalServerErrorException,
} from "@nestjs/common";
import { ConfigService } from "@nestjs/config";
import OpenAI from "openai";
import * as path from "path";
import * as mammoth from "mammoth";
import { PDFParse } from "pdf-parse";

const DEFAULT_TOPIC_MODEL = "gpt-5";
const DEFAULT_SUMMARY_MODEL = "gpt-5";
const TOPIC_SCHEMA = {
  name: "topic_segmentation",
  schema: {
    type: "object",
    additionalProperties: false,
    required: ["topics"],
    properties: {
      topics: {
        type: "array",
        minItems: 1,
        items: {
          type: "object",
          additionalProperties: false,
          required: ["title", "content"],
          properties: {
            title: { type: "string" },
            content: { type: "string" },
          },
        },
      },
    },
  },
  strict: true,
} as const;

@Injectable()
export class DocumentTopicsService {
  private readonly openai: OpenAI;
  private readonly openaiApiKey?: string;
  private readonly topicModel: string;
  private readonly fallbackTopicModels: string[];
  private readonly summaryModel: string;
  private readonly fallbackSummaryModels: string[];

  constructor(private readonly config: ConfigService) {
    const apiKey =
      this.config.get<string>("OPENAI_API_KEY") || process.env.OPENAI_API_KEY;
    this.openaiApiKey = apiKey;
    this.openai = new OpenAI({ apiKey });
    this.topicModel =
      this.config.get<string>("OPENAI_TOPIC_MODEL") ||
      process.env.OPENAI_TOPIC_MODEL ||
      DEFAULT_TOPIC_MODEL;
    const fallbackFromEnv =
      this.config.get<string>("OPENAI_TOPIC_MODEL_FALLBACKS") ||
      process.env.OPENAI_TOPIC_MODEL_FALLBACKS ||
      "gpt-4.1";
    this.fallbackTopicModels = fallbackFromEnv
      .split(",")
      .map((v) => v.trim())
      .filter((v) => !!v && v !== this.topicModel);
    this.summaryModel =
      this.config.get<string>("OPENAI_SUMMARY_MODEL") ||
      process.env.OPENAI_SUMMARY_MODEL ||
      DEFAULT_SUMMARY_MODEL;
    const fallbackSummaryFromEnv =
      this.config.get<string>("OPENAI_SUMMARY_MODEL_FALLBACKS") ||
      process.env.OPENAI_SUMMARY_MODEL_FALLBACKS ||
      "gpt-4.1";
    this.fallbackSummaryModels = fallbackSummaryFromEnv
      .split(",")
      .map((v) => v.trim())
      .filter((v) => !!v && v !== this.summaryModel);
  }

  async extractPlainText(
    buffer: Buffer,
    mimeType: string,
    originalName: string,
  ): Promise<string> {
    const ext = path.extname(originalName || "").toLowerCase();
    const mime = (mimeType || "").toLowerCase();
    console.log("[DocumentTopics] extracting text", {
      originalName,
      mimeType: mime || "unknown",
      extension: ext || "none",
      sizeBytes: buffer.length,
    });

    if (mime === "application/pdf" || ext === ".pdf") {
      console.log("[DocumentTopics] parser selected: pdf");
      const parser = new PDFParse({ data: buffer });
      try {
        const textResult = await parser.getText();
        console.log("[DocumentTopics] pdf extraction done", {
          extractedTextLength: (textResult.text ?? "").length,
        });
        return (textResult.text ?? "").trim();
      } finally {
        await parser.destroy();
      }
    }

    if (
      mime ===
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
      ext === ".docx"
    ) {
      console.log("[DocumentTopics] parser selected: docx (mammoth)");
      const result = await mammoth.extractRawText({ buffer });
      console.log("[DocumentTopics] docx extraction done", {
        extractedTextLength: (result.value ?? "").length,
      });
      return (result.value ?? "").trim();
    }

    if (
      mime.startsWith("text/") ||
      [
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".json",
        ".xml",
        ".html",
        ".htm",
        ".rtf",
      ].includes(ext)
    ) {
      console.log("[DocumentTopics] parser selected: utf8 text");
      return buffer.toString("utf8").trim();
    }

    if (mime === "application/msword" || ext === ".doc") {
      throw new BadRequestException(
        "Legacy .doc format is not supported. Please save as .docx or PDF and upload again.",
      );
    }

    throw new BadRequestException(
      `Unsupported file type (“${mime || ext || "unknown"}”). Supported: PDF, DOCX, and common text-based formats (TXT, MD, CSV, JSON, XML, HTML).`,
    );
  }

  async segmentByTopicsInOrder(plainText: string): Promise<
    { title: string; content: string }[]
  > {
    if (!plainText) {
      throw new BadRequestException("No text could be extracted from the file.");
    }
    if (!this.openaiApiKey) {
      throw new InternalServerErrorException(
        "OPENAI_API_KEY is missing. Add it to your environment before analyzing topics.",
      );
    }

    const candidateModels = [this.topicModel, ...this.fallbackTopicModels];
    console.log("[DocumentTopics] segmenting topics", {
      inputTextLength: plainText.length,
      model: this.topicModel,
      fallbacks: this.fallbackTopicModels,
    });

    const systemInstruction = `
You segment instructional or narrative documents into topics for downstream processing.

Rules:
1. Return strictly valid JSON that conforms to the provided schema.
2. Shape: {"topics":[{"title":"string","content":"string"}, ...]}
3. Preserve the document's topic order exactly as the reader encounters it (top to bottom), but skip navigational front matter as specified in rule 9.
4. When the document uses headings (numbered sections, chapter titles, bold lead lines, etc.), use those as "title". Otherwise invent short, accurate section titles.
5. Each "content" is the continuous body text belonging to that topic only — no title repeated inside content unless it also appears in the original body.
6. Do not merge unrelated sections; do not reorder; do not drop substantive paragraphs (except entire sections excluded by rule 9).
7. If the document is one continuous piece with no clear divisions, return a single topic with a concise title and the full text as content.
8. Normalize whitespace lightly (collapse excessive blank lines) but do not paraphrase or summarize.
9. Omit "Table of Contents", "Contents", "Index", and similar outline-only blocks entirely: do not emit a topic whose title or content is mainly a listing of section names, chapter titles, or page numbers. Start topics from the first substantive section (e.g. Introduction, Chapter 1, or the first real body heading after the TOC). Do not paste TOC lines into any other topic's content as filler or context.
`.trim();

    let parsed: unknown;
    let lastParseError: unknown;
    let lastModelError: unknown;
    let usedModel = this.topicModel;

    for (const modelName of candidateModels) {
      usedModel = modelName;
      for (let attempt = 1; attempt <= 2; attempt++) {
        let response: unknown;
        try {
          response = await this.openai.responses.create({
            model: modelName,
            temperature: 0.1,
            max_output_tokens: 8192,
            input: [
              { role: "system", content: systemInstruction },
              { role: "user", content: plainText },
            ],
            text: {
              format: {
                type: "json_schema",
                ...TOPIC_SCHEMA,
              },
            },
          });
        } catch (err) {
          lastModelError = err;
          console.error("[DocumentTopics] model request failed", {
            model: modelName,
            attempt,
            error: err instanceof Error ? err.message : String(err),
          });
          break;
        }

        const rawText =
          (response as { output_text?: string })?.output_text ??
          (
            response as {
              output?: { content?: { text?: string; type?: string }[] }[];
            }
          )?.output
            ?.flatMap((item) => item.content ?? [])
            .filter((part) => part.type === "output_text" && !!part.text)
            .map((part) => part.text ?? "")
            .join("") ??
          "";

        console.log("[DocumentTopics] model response received", {
          model: modelName,
          attempt,
          rawLength: rawText.length,
        });

        try {
          parsed = JSON.parse(rawText);
          break;
        } catch (err) {
          lastParseError = err;
          if (attempt === 2) {
            break;
          }
        }
      }
      if (parsed) {
        break;
      }
    }

    if (!parsed) {
      const modelErrorMessage =
        lastModelError instanceof Error ? lastModelError.message : "";
      console.error("[DocumentTopics] failed to parse topic JSON", {
        model: usedModel,
        error:
          lastParseError instanceof Error
            ? lastParseError.message
            : String(lastParseError),
        modelErrorMessage,
      });

      if (
        modelErrorMessage.includes("model") &&
        (modelErrorMessage.includes("not found") ||
          modelErrorMessage.includes("do not have access"))
      ) {
        throw new InternalServerErrorException(
          `Topic segmentation model is unavailable (${usedModel}). Set OPENAI_TOPIC_MODEL to a model your key can access, for example gpt-4.1.`,
        );
      }
      throw new InternalServerErrorException(
        "Topic segmentation failed before a valid JSON response was produced.",
      );
    }

    const topics = (parsed as { topics?: unknown })?.topics;
    if (!Array.isArray(topics)) {
      throw new InternalServerErrorException(
        'Topic segmentation response missing a "topics" array.',
      );
    }

    const out: { title: string; content: string }[] = [];
    for (let i = 0; i < topics.length; i++) {
      const row = topics[i] as { title?: unknown; content?: unknown };
      const title = typeof row?.title === "string" ? row.title.trim() : "";
      const content = typeof row?.content === "string" ? row.content.trim() : "";
      if (!title && !content) continue;
      out.push({
        title: title || `Section ${i + 1}`,
        content,
      });
    }

    if (out.length === 0) {
      throw new InternalServerErrorException(
        "Topic segmentation produced no sections.",
      );
    }
    console.log("[DocumentTopics] topic segmentation complete", {
      topicCount: out.length,
    });

    return out;
  }

  async analyzeUploadedDocument(
    buffer: Buffer,
    mimeType: string,
    originalName: string,
  ) {
    console.log("[DocumentTopics] analysis started", { originalName });
    const text = await this.extractPlainText(buffer, mimeType, originalName);
    const topics = await this.segmentByTopicsInOrder(text);
    console.log("[DocumentTopics] analysis finished", {
      originalName,
      extractedTextLength: text.length,
      topicCount: topics.length,
    });
    return {
      topics,
      extractedTextLength: text.length,
      originalFileName: originalName,
    };
  }

  async summarizePlainText(plainText: string): Promise<string> {
    if (!plainText) {
      throw new BadRequestException("No text could be extracted from the file.");
    }
    if (!this.openaiApiKey) {
      throw new InternalServerErrorException(
        "OPENAI_API_KEY is missing. Add it to your environment before generating a summary.",
      );
    }

    const candidateModels = [this.summaryModel, ...this.fallbackSummaryModels];
    const systemInstruction = `
You produce a detailed instructional summary of an entire document for use in educational or demonstration video generation.

Rules:
1. Return plain text only (no JSON, no markdown, no bullet symbols).
2. Cover the full document — do not skip sections, chapters, or topics. Every major section must be represented.
3. For each section or topic, write a dedicated paragraph that explains: the purpose of that section, the key steps or concepts involved, any warnings, safety notes, or prerequisites, and the expected outcome or result.
4. Preserve the original document's order and structure so the video can follow the same sequence.
5. Use clear, action-oriented language written as if narrating or explaining to a viewer watching a demonstration.
6. Remove only exact duplicate content — near-similar steps that appear in different contexts should still be described separately if they serve a different purpose.
7. Do not invent, assume, or expand on facts not present in the document.
8. Do not summarize multiple distinct topics into one paragraph — keep each topic or section as its own focused block.
9. Length must reflect the document's depth. A short document (1–5 pages) should produce 400–700 words. A medium document (6–20 pages) should produce 700–1500 words. A long document such as a user manual or technical guide (20+ pages) should produce 1500–3000 words or more as needed to fully represent all content.
10. Terminology must remain accurate and consistent with the source document throughout.
`.trim();

    let finalSummary = "";
    let usedModel = this.summaryModel;
    let lastError: unknown;

    for (const modelName of candidateModels) {
      usedModel = modelName;
      try {
        const response = await this.openai.responses.create({
          model: modelName,
          temperature: 0.2,
          max_output_tokens: 3500,
          input: [
            { role: "system", content: systemInstruction },
            {
              role: "user",
              content: `Summarize this document for cinematic educational scene generation:\n\n${plainText}`,
            },
          ],
        });

        finalSummary =
          (response as { output_text?: string })?.output_text?.trim() ?? "";

        if (finalSummary) {
          break;
        }
      } catch (err) {
        lastError = err;
      }
    }

    if (!finalSummary) {
      const modelErrorMessage = lastError instanceof Error ? lastError.message : "";
      if (
        modelErrorMessage.includes("model") &&
        (modelErrorMessage.includes("not found") ||
          modelErrorMessage.includes("do not have access"))
      ) {
        throw new InternalServerErrorException(
          `Summary model is unavailable (${usedModel}). Set OPENAI_SUMMARY_MODEL to a model your key can access, for example gpt-4.1.`,
        );
      }
      throw new InternalServerErrorException(
        "Document summarization failed before a usable response was produced.",
      );
    }

    return finalSummary;
  }

  async summarizeUploadedDocument(
    buffer: Buffer,
    mimeType: string,
    originalName: string,
  ) {
    const text = await this.extractPlainText(buffer, mimeType, originalName);
    const summary = await this.summarizePlainText(text);
    return {
      originalFileName: originalName,
      extractedTextLength: text.length,
      summaryTextLength: summary.length,
      summary,
    };
  }

  async analyzeTopics(file: Express.Multer.File | undefined) {
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
    return await this.analyzeUploadedDocument(
      file.buffer,
      file.mimetype,
      file.originalname,
    );
  }
}
