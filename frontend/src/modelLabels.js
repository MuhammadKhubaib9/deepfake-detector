// Friendly display names for the per-model scores shown in verdicts.
// Backend keys stay stable (cnn, efficientnet, vit, vit_l14);
// only the presentation name changes here.
export const MODEL_LABELS = {
  cnn: "XceptionNet",
  efficientnet: "EfficientNet-B3",
  vit: "ViT model",
  vit_l14: "ViT-L/14",
};

export function formatModelScores(scores = {}) {
  return Object.entries(scores)
    .map(([k, v]) => (MODEL_LABELS[k] || k) + " " + (+v * 100).toFixed(0) + "%")
    .join(" · ");
}