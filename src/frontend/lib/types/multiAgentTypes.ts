export interface QueryUnderstanding {
  query_type: string;
  key_terms: string[];
}

export interface Retrieval {
  document_count: number;
}

export interface Evaluation {
  relevance?: number;
  grounding?: number;
  completeness?: number;
  coherence?: number;
  conciseness?: number;
  [key: string]: number | undefined;
}

export interface MultiAgentMetadata {
  query_understanding: QueryUnderstanding;
  retrieval: Retrieval;
  evaluation: Evaluation;
  processing_stages: string[];
}
