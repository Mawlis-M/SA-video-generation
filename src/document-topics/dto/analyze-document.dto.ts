export class DocumentTopicSectionDto {
  /** Topic or section title (from the document heading when possible). */
  title: string;

  /** Body text for this topic, in original order. */
  content: string;
}

export class AnalyzeDocumentResponseDto {
  topics: DocumentTopicSectionDto[];

  /** Character length of raw extracted text (before topic segmentation). */
  extractedTextLength: number;

  originalFileName?: string;
}
