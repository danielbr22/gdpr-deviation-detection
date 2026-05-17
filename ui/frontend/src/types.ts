export interface LogEntry {
  filename: string;
  timestamp: string | null;
  has_results: boolean;
  size_bytes: number;
}

export interface GoldResult {
  id: string;
  deviation_type: string;
  articles: number[];
  detected: boolean;
}

export interface Metrics {
  precision: number;
  recall: number;
  f1: number;
  tp: number;
  fp: number;
  fn: number;
}

export interface UseCaseResult {
  use_case: string;
  gold_results: GoldResult[];
  type_metrics: Record<string, Metrics>;
  overall: Metrics;
}

export interface Results {
  results: UseCaseResult[];
}
