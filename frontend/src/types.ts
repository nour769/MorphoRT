export type DataSource = "mock" | "real";

export type PageKey =
  "home" | "analysis" | "growth" | "radiotherapy" | "validation" | "methods";

export interface MeasurementResult {
  volume_cm3: number;
  surface_cm2: number;
  sphericity: number;
}

export interface PackedVolume {
  encoding: "zlib-u8" | "zlib-packbits-little";
  data: string;
}

export interface AnalysisResult {
  data_source: DataSource;
  shape: [number, number, number];
  voxel_size_mm: [number, number, number];
  image: PackedVolume;
  mask: PackedVolume;
  measurements: MeasurementResult;
  parameters: Record<string, string | number>;
}

export interface CurvePoint {
  day: number;
  volume_cm3: number;
  lower_cm3: number;
  upper_cm3: number;
}

export interface GrowthResult {
  data_source: DataSource;
  model: "gompertz" | "logistic";
  r: number;
  K: number;
  ci_r: [number, number];
  ci_K: [number, number];
  curve: CurvePoint[];
}

export interface TrajectoryPoint {
  fraction: number;
  volume_cm3: number;
  dose_gy: number;
  bed_gy: number;
}

export interface RadiotherapyResult {
  data_source: DataSource;
  simulation: true;
  fixed: TrajectoryPoint[];
  adaptive: TrajectoryPoint[];
}

export interface ValidationResult {
  data_source: DataSource;
  available: boolean;
  message?: string;
  count: number;
  dice: number[];
  volumes: Array<{
    cas: string;
    volume_cm3: number;
    volume_expert_cm3: number;
  }>;
  mean: number | null;
  median: number | null;
  std: number | null;
}

export interface GrowthPointInput {
  day: number;
  volume_cm3: number;
}

export interface GrowthParameters {
  volume_cm3: number;
  r: number;
  K: number;
}
