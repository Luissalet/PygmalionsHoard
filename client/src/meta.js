// Vocabulary shared with the backend. Labels live in i18n.js.
export const ARTIFACT_KINDS = ["base", "adapter", "merged", "ctx_variant", "gguf", "imatrix", "ollama"];
export const JOB_KINDS = ["download", "dataset_build", "train", "merge_lora", "merge_models", "convert", "imatrix", "quantize", "perplexity", "ctx_extend", "publish", "evaluate"];
export const QUANT_TYPES = [
  { id: "Q8_0", bits: 8.5 },
  { id: "Q6_K", bits: 6.56 },
  { id: "Q5_K_M", bits: 5.69 },
  { id: "Q4_K_M", bits: 4.85 },
  { id: "IQ4_XS", bits: 4.3, imatrix: true },
  { id: "Q3_K_M", bits: 3.9 },
  { id: "IQ3_M", bits: 3.66, imatrix: true },
];
export const OUT_TYPES = ["f16", "bf16", "q8_0"];
export const MERGE_METHODS = ["linear", "slerp", "ties", "dare"];
export const INTENTS = ["", "dataset", "style", "code", "context", "general", "smoke", "quant"];
export const CTX_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072, 262144];

export const JOB_STATE_CLASS = {
  planned: "", queued: "", waiting_gpu: "chip-amber", running: "chip-accent", done: "chip-ok", failed: "chip-danger", cancelled: "", interrupted: "chip-amber",
};
export const VERDICT_CLASS = { better: "chip-ok", worse: "chip-danger", no_clear_difference: "chip-amber", no_data: "", unknown: "" };

// 24x24 stroke icon paths per artifact kind
export const KIND_ICON = {
  base: "M4 7l8-4 8 4v10l-8 4-8-4zM4 7l8 4 8-4M12 11v10",
  adapter: "M4 12h4l2-5 4 10 2-5h4",
  merged: "M6 4v6a6 6 0 006 6 6 6 0 006-6V4M12 16v5",
  ctx_variant: "M4 12h16M4 12l4-4M4 12l4 4M20 12l-4-4M20 12l-4 4",
  gguf: "M7 3h8l4 4v14H7zM15 3v4h4M10 13h6M10 17h6",
  imatrix: "M4 4h16v16H4zM4 9h16M4 14h16M9 4v16M14 4v16",
  ollama: "M12 3a5 5 0 015 5v2a5 5 0 01-10 0V8a5 5 0 015-5zM7 21c0-3 2-5 5-5s5 2 5 5",
};
