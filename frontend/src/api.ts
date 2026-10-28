import type {
  AnalysisResult,
  DataSource,
  GrowthPointInput,
  GrowthParameters,
  GrowthResult,
  RadiotherapyResult,
  ValidationResult,
} from "./types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

interface SourceResponse {
  data_source: DataSource;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch (error) {
    throw new Error(
      error instanceof Error
        ? `API inaccessible : ${error.message}`
        : "API inaccessible.",
    );
  }
  let body: unknown;
  try {
    body = await response.json();
  } catch (error) {
    throw new Error(
      error instanceof Error
        ? `Réponse API invalide : ${error.message}`
        : "Réponse API invalide.",
    );
  }
  if (!response.ok) {
    const errorDetail =
      typeof body === "object" &&
      body !== null &&
      "detail" in body &&
      body.detail;
    const detail =
      typeof errorDetail === "string"
        ? errorDetail
        : Array.isArray(errorDetail)
          ? errorDetail
              .map((item) =>
                typeof item === "object" &&
                item !== null &&
                "message" in item &&
                typeof item.message === "string"
                  ? item.message
                  : "Paramètre invalide.",
              )
              .join(" ")
          : `Erreur API (${response.status}).`;
    throw new Error(detail);
  }
  return body as T;
}

export async function getDataSource(): Promise<DataSource> {
  const result = await request<SourceResponse>("/api/health");
  return result.data_source;
}

export async function analyzeVolume(
  file: File,
  t1ce?: File,
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("flair", file);
  if (t1ce) formData.append("t1ce", t1ce);
  return request<AnalysisResult>("/api/analyze", {
    method: "POST",
    body: formData,
  });
}

export async function fitGrowth(
  points: GrowthPointInput[],
  model: "gompertz" | "logistic",
): Promise<GrowthResult> {
  return request<GrowthResult>("/api/growth", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      volumes_by_day: points,
    }),
  });
}

export async function simulateRadiotherapy(
  params: GrowthParameters & {
    alpha: number;
    beta: number;
    bed_max: number;
    n_fractions: number;
  },
): Promise<RadiotherapyResult> {
  return request<RadiotherapyResult>("/api/radiotherapy", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

export async function getValidation(): Promise<ValidationResult> {
  return request<ValidationResult>("/api/validation");
}
