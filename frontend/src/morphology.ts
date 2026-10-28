export type BinaryImage = number[][];
export type MorphologyOperation = "dilation" | "erosion" | "opening" | "closing";

export const SAMPLE_IMAGE: BinaryImage = Array.from({ length: 17 }, (_, row) =>
  Array.from({ length: 17 }, (_, column) => {
    const square = row >= 4 && row <= 12 && column >= 4 && column <= 12;
    const hole = row === 8 && column === 8;
    const noise = (row === 2 && column === 2) || (row === 14 && column === 14);
    return Number((square && !hole) || noise);
  }),
);

function applyOneStep(
  image: BinaryImage,
  size: number,
  operation: "dilation" | "erosion",
): BinaryImage {
  const radius = Math.floor(size / 2);

  return image.map((row, y) =>
    row.map((_, x) => {
      let foundForeground = false;
      let everyNeighborIsForeground = true;

      for (let dy = -radius; dy <= radius; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          const neighborY = y + dy;
          const neighborX = x + dx;
          const isForeground =
            neighborY >= 0 &&
            neighborY < image.length &&
            neighborX >= 0 &&
            neighborX < row.length &&
            image[neighborY][neighborX] === 1;

          foundForeground ||= isForeground;
          everyNeighborIsForeground &&= isForeground;
        }
      }

      return Number(
        operation === "dilation" ? foundForeground : everyNeighborIsForeground,
      );
    }),
  );
}

export function applyMorphology(
  image: BinaryImage,
  operation: MorphologyOperation,
  size: number,
): BinaryImage {
  if (!Number.isInteger(size) || size < 1 || size % 2 === 0) {
    throw new Error("La taille du voisinage doit être un entier impair positif.");
  }
  if (
    image.length === 0 ||
    image[0].length === 0 ||
    image.some((row) => row.length !== image[0].length)
  ) {
    throw new Error("L’image binaire doit être rectangulaire et non vide.");
  }

  switch (operation) {
    case "dilation":
      return applyOneStep(image, size, "dilation");
    case "erosion":
      return applyOneStep(image, size, "erosion");
    case "opening":
      return applyOneStep(applyOneStep(image, size, "erosion"), size, "dilation");
    case "closing":
      return applyOneStep(applyOneStep(image, size, "dilation"), size, "erosion");
  }
}
