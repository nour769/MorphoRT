export function formatMetric(value: number, maximumFractionDigits = 2): string {
  return new Intl.NumberFormat("fr-FR", {
    maximumFractionDigits,
  }).format(value);
}

export function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "Une erreur inattendue est survenue.";
}

export async function decompressBase64(data: string): Promise<Uint8Array> {
  const binary = atob(data);
  const compressed = Uint8Array.from(binary, (character) =>
    character.charCodeAt(0),
  );
  const stream = new Blob([compressed])
    .stream()
    .pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
