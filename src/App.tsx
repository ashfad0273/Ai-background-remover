import { useEffect, useRef, useState, useCallback } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────
type AppStatus =
  | "idle"
  | "loading-model"
  | "model-ready"
  | "processing"
  | "done"
  | "error";

interface ProgressInfo {
  file: string;
  loaded: number;
  total: number;
  percent: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Checkerboard SVG background (data-url) used in the result panel to visually
 * indicate transparency — the industry-standard pattern.
 */
const CHECKER_BG = `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24'%3E%3Crect width='12' height='12' fill='%23cbd5e1'/%3E%3Crect x='12' y='12' width='12' height='12' fill='%23cbd5e1'/%3E%3Crect x='12' y='0' width='12' height='12' fill='%23f1f5f9'/%3E%3Crect x='0' y='12' width='12' height='12' fill='%23f1f5f9'/%3E%3C/svg%3E")`;

// ─────────────────────────────────────────────────────────────────────────────
// Helper utilities
// ─────────────────────────────────────────────────────────────────────────────

/** Format a byte count into a human-readable string (e.g. "12.4 MB") */
function fmtBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(k)), sizes.length - 1);
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

/** Strip the file extension from a filename */
function stripExt(name: string): string {
  return name.replace(/\.[^/.]+$/, "") || "image";
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

/** Animated SVG spinner */
function Spinner({
  size = 28,
  color = "#818cf8",
}: {
  size?: number;
  color?: string;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="animate-spin"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="10" strokeOpacity="0.2" />
      <path d="M12 2a10 10 0 0 1 10 10" />
    </svg>
  );
}

/** Ionicon wrapper with correct typings */
function Ion({
  name,
  size = 20,
  color,
  className = "",
}: {
  name: string;
  size?: number;
  color?: string;
  className?: string;
}) {
  // We use dangerouslySetInnerHTML as a workaround because Ionicons are
  // registered as custom elements and TypeScript doesn't know about them.
  return (
    <span
      className={`inline-flex items-center justify-center ${className}`}
      style={{ lineHeight: 0, color }}
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{
        __html: `<ion-icon name="${name}" style="font-size:${size}px;color:${color ?? "currentColor"}"></ion-icon>`,
      }}
    />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// App
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  // ── State ──────────────────────────────────────────────────────────────────
  const [status, setStatus] = useState<AppStatus>("idle");
  const [statusText, setStatusText] = useState("Initializing…");
  const [errorMsg, setErrorMsg] = useState("");
  const [progress, setProgress] = useState<ProgressInfo | null>(null);

  const [originalSrc, setOriginalSrc] = useState<string | null>(null);
  const [resultSrc, setResultSrc] = useState<string | null>(null);
  const [fileName, setFileName] = useState("");
  const [isDragOver, setIsDragOver] = useState(false);

  // ── Refs ───────────────────────────────────────────────────────────────────
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const modelRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const processorRef = useRef<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Track object URLs so we can revoke them and avoid memory leaks
  const prevObjectUrlRef = useRef<string | null>(null);

  // ── Derived state ──────────────────────────────────────────────────────────
  const isModelLoading = status === "loading-model";
  const isProcessing = status === "processing";
  const isReady = status === "model-ready" || status === "done";
  const isDone = status === "done";
  const showOverlay = isModelLoading || isProcessing;

  // ── Load model on mount ────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;

    async function loadModel() {
      try {
        setStatus("loading-model");
        setStatusText("Downloading AI Model…");

        /**
         * Dynamic import keeps the heavy onnxruntime-web out of the initial
         * bundle chunk, so the page shell renders immediately.
         */
        const { AutoModel, AutoProcessor, env } = await import(
          "@xenova/transformers"
        );

        // ── Transformers.js environment configuration ──────────────────────
        // Only use remote models (HF Hub); disable local filesystem lookups.
        env.allowLocalModels = false;
        // Proxy the ONNX WASM backend to a worker so the UI thread stays
        // responsive during heavy matrix math.
        env.backends.onnx.wasm.proxy = true;

        // ── Progress callback ──────────────────────────────────────────────
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const onProgress = (p: any) => {
          switch (p.status) {
            case "initiate":
              setStatusText(`Fetching: ${p.file ?? "model"}…`);
              break;
            case "progress":
              if (p.total && p.total > 0) {
                setProgress({
                  file: p.file ?? "",
                  loaded: p.loaded ?? 0,
                  total: p.total ?? 0,
                  percent: Math.min(
                    Math.round(((p.loaded ?? 0) / p.total) * 100),
                    100
                  ),
                });
              }
              break;
            case "done":
              setProgress(null);
              setStatusText("Loading components…");
              break;
            case "ready":
              setStatusText("Almost ready…");
              break;
          }
        };

        // ── Load the RMBG-1.4 model ────────────────────────────────────────
        // The 8-bit quantized ONNX version is ~45 MB and is cached by the
        // browser's Cache API after the first download.
        const model = await AutoModel.from_pretrained("briaai/RMBG-1.4", {
          config: { model_type: "custom" },
          progress_callback: onProgress,
        });

        if (cancelled) return;
        setStatusText("Loading image processor…");

        // ── Load the feature extractor / processor ─────────────────────────
        // We specify the config explicitly because auto-detection sometimes
        // falls back to incorrect defaults for this custom model type.
        const processor = await AutoProcessor.from_pretrained("briaai/RMBG-1.4", {
          config: {
            do_normalize: true,
            do_pad: false,
            do_rescale: true,
            do_resize: true,
            image_mean: [0.5, 0.5, 0.5],
            feature_extractor_type: "ImageFeatureExtractor",
            image_std: [1, 1, 1],
            resample: 2,
            rescale_factor: 0.00392156862745098, // 1/255
            size: { width: 1024, height: 1024 },
          },
        });

        if (cancelled) return;

        modelRef.current = model;
        processorRef.current = processor;
        setStatus("model-ready");
        setStatusText("Model ready!");
        setProgress(null);
      } catch (err: unknown) {
        if (cancelled) return;
        console.error("[BG Eraser] Model load error:", err);
        setStatus("error");
        setErrorMsg(
          err instanceof Error
            ? `${err.message}`
            : "Failed to load the AI model. Please check your internet connection and refresh the page."
        );
      }
    }

    loadModel();
    return () => { cancelled = true; };
  }, []);

  // ── Core processing logic ──────────────────────────────────────────────────
  const processImage = useCallback(async (file: File) => {
    if (!modelRef.current || !processorRef.current) return;
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file (JPG, PNG, WEBP, etc.).");
      return;
    }

    try {
      setStatus("processing");
      setResultSrc(null);
      setFileName(stripExt(file.name));

      // ── Show original preview immediately ──────────────────────────────
      // Revoke any previously created object URL to free memory.
      if (prevObjectUrlRef.current) {
        URL.revokeObjectURL(prevObjectUrlRef.current);
      }
      const objectUrl = URL.createObjectURL(file);
      prevObjectUrlRef.current = objectUrl;
      setOriginalSrc(objectUrl);

      setStatusText("Reading image data…");
      // RawImage is a lightweight in-memory image class provided by Transformers.js
      const { RawImage } = await import("@xenova/transformers");
      const rawImage = await RawImage.fromBlob(file);

      setStatusText("Preprocessing for AI model…");
      // The processor resizes + normalises the image to 1024×1024
      const { pixel_values } = await processorRef.current(rawImage);

      setStatusText("Running AI segmentation…");
      // Forward pass — output is a single-channel float tensor (sigmoid output)
      const { output } = await modelRef.current({ input: pixel_values });

      setStatusText("Applying transparency mask…");
      // Convert float [0,1] → uint8 [0,255] and resize to match original dimensions
      const mask = await RawImage.fromTensor(
        output[0].mul(255).to("uint8")
      ).resize(rawImage.width, rawImage.height);

      // ── Apply mask via Canvas ──────────────────────────────────────────
      // This is the cleanest approach: draw the original image, then overwrite
      // every pixel's alpha channel with the corresponding mask value.
      const canvas = canvasRef.current!;
      canvas.width = rawImage.width;
      canvas.height = rawImage.height;
      const ctx = canvas.getContext("2d")!;

      // 1. Paint original image
      ctx.drawImage(rawImage.toCanvas() as CanvasImageSource, 0, 0);

      // 2. Read pixel data and update alpha channel from mask
      const imgData = ctx.getImageData(0, 0, rawImage.width, rawImage.height);
      for (let i = 0; i < mask.data.length; i++) {
        imgData.data[4 * i + 3] = mask.data[i]; // R G B [A]
      }
      ctx.putImageData(imgData, 0, 0);

      // 3. Export as lossless transparent PNG
      const pngDataUrl = canvas.toDataURL("image/png");
      setResultSrc(pngDataUrl);
      setStatus("done");
      setStatusText("Background removed!");
    } catch (err: unknown) {
      console.error("[BG Eraser] Processing error:", err);
      setStatus("error");
      setErrorMsg(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred while processing your image. Please try again."
      );
    }
  }, []);

  // ── File handling ──────────────────────────────────────────────────────────
  const openFilePicker = useCallback(() => {
    if (!isReady) return;
    fileInputRef.current?.click();
  }, [isReady]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processImage(file);
    e.target.value = ""; // Allow re-selecting the same file
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && isReady) processImage(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (isReady) setIsDragOver(true);
  };

  const handleDragLeave = () => setIsDragOver(false);

  // ── Download handler ───────────────────────────────────────────────────────
  const handleDownload = () => {
    if (!resultSrc) return;
    const link = document.createElement("a");
    link.href = resultSrc;
    link.download = `${fileName}_no-bg.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // ── Reset / new image ──────────────────────────────────────────────────────
  const handleReset = () => {
    if (prevObjectUrlRef.current) {
      URL.revokeObjectURL(prevObjectUrlRef.current);
      prevObjectUrlRef.current = null;
    }
    setOriginalSrc(null);
    setResultSrc(null);
    setFileName("");
    setStatus("model-ready");
    setStatusText("Model ready!");
    setErrorMsg("");
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-[#0d1224] to-indigo-950 text-white flex flex-col select-none">

      {/* ── Hidden processing canvas ────────────────────────────────────── */}
      <canvas ref={canvasRef} className="hidden" aria-hidden="true" />

      {/* ── Hidden file input ───────────────────────────────────────────── */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileChange}
        aria-label="Upload image"
      />

      {/* ══════════════════════════════════════════════════════════════════
          LOADING / PROCESSING OVERLAY
      ══════════════════════════════════════════════════════════════════ */}
      {showOverlay && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/75 backdrop-blur-md"
          role="status"
          aria-live="polite"
        >
          <div className="bg-slate-900 border border-slate-700/60 rounded-3xl px-8 py-8 flex flex-col items-center gap-5 shadow-2xl shadow-black/50 max-w-sm w-full mx-4">
            {/* Animated icon ring */}
            <div className="relative flex items-center justify-center">
              <div className="absolute w-20 h-20 rounded-full bg-indigo-500/10 animate-pulse" />
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-indigo-900 to-slate-800 border border-indigo-500/30 flex items-center justify-center shadow-inner">
                {isModelLoading ? (
                  <Ion name="cloud-download-outline" size={30} color="#818cf8" />
                ) : (
                  <Ion name="color-wand-outline" size={30} color="#818cf8" />
                )}
              </div>
              <div className="absolute -bottom-1 -right-1">
                <Spinner size={22} color="#818cf8" />
              </div>
            </div>

            {/* Status text */}
            <div className="text-center space-y-1">
              <p className="text-white font-semibold text-[15px]">{statusText}</p>
              {progress && (
                <p className="text-slate-400 text-xs">
                  {fmtBytes(progress.loaded)} / {fmtBytes(progress.total)}
                </p>
              )}
            </div>

            {/* Progress bar */}
            {progress && (
              <div className="w-full space-y-1.5">
                <div className="flex justify-between text-[11px] text-slate-500">
                  <span className="truncate max-w-[200px] font-mono">{progress.file}</span>
                  <span className="font-semibold text-indigo-400">{progress.percent}%</span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-indigo-500 via-violet-500 to-purple-500 transition-[width] duration-300 ease-out"
                    style={{ width: `${progress.percent}%` }}
                  />
                </div>
              </div>
            )}

            {/* Helper text */}
            <p className="text-slate-500 text-xs text-center leading-relaxed max-w-xs">
              {isModelLoading
                ? "The AI model (~45 MB) downloads once and is cached in your browser — future sessions load instantly."
                : "Processing happens entirely on your device. No data is sent anywhere."}
            </p>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════
          HEADER
      ══════════════════════════════════════════════════════════════════ */}
      <header className="sticky top-0 z-40 border-b border-white/5 bg-slate-950/70 backdrop-blur-xl">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          {/* Brand */}
          <div className="flex items-center gap-3 shrink-0">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/30">
              <Ion name="cut-outline" size={18} color="white" />
            </div>
            <div className="leading-tight">
              <span className="font-bold text-white text-[17px] tracking-tight">BG Eraser</span>
              <span className="hidden sm:inline text-slate-500 text-xs ml-2">
                AI Background Remover
              </span>
            </div>
          </div>

          {/* Center pills */}
          <div className="hidden md:flex items-center gap-1.5">
            {[
              { icon: "lock-closed-outline", label: "100% Private", color: "#34d399" },
              { icon: "cash-outline", label: "Always Free", color: "#fbbf24" },
              { icon: "server-outline", label: "No Server", color: "#818cf8" },
            ].map(({ icon, label, color }) => (
              <span
                key={label}
                className="flex items-center gap-1 px-2.5 py-1 rounded-full bg-slate-800/70 border border-slate-700/50 text-slate-400 text-xs"
              >
                <Ion name={icon} size={11} color={color} />
                {label}
              </span>
            ))}
          </div>

          {/* Status badge */}
          <div
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border shrink-0 transition-all duration-300 ${
              status === "error"
                ? "bg-red-950/60 border-red-700/40 text-red-400"
                : isModelLoading
                ? "bg-amber-950/60 border-amber-700/40 text-amber-400"
                : isProcessing
                ? "bg-indigo-950/60 border-indigo-700/40 text-indigo-400"
                : isDone
                ? "bg-emerald-950/60 border-emerald-700/40 text-emerald-400"
                : isReady
                ? "bg-emerald-950/60 border-emerald-700/40 text-emerald-400"
                : "bg-slate-800 border-slate-700 text-slate-500"
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                status === "error"
                  ? "bg-red-500"
                  : isModelLoading
                  ? "bg-amber-400 animate-pulse"
                  : isProcessing
                  ? "bg-indigo-400 animate-pulse"
                  : isReady || isDone
                  ? "bg-emerald-400"
                  : "bg-slate-600"
              }`}
            />
            {status === "error"
              ? "Error"
              : isModelLoading
              ? "Loading Model…"
              : isProcessing
              ? "Processing…"
              : isDone
              ? "Done!"
              : isReady
              ? "AI Ready"
              : "Idle"}
          </div>
        </div>
      </header>

      {/* ══════════════════════════════════════════════════════════════════
          MAIN CONTENT
      ══════════════════════════════════════════════════════════════════ */}
      <main className="flex-1 max-w-6xl mx-auto px-4 sm:px-6 py-10 w-full">

        {/* ── Error Banner ────────────────────────────────────────────── */}
        {status === "error" && (
          <div className="mb-8 bg-red-950/40 border border-red-700/40 rounded-2xl p-5 flex items-start gap-4">
            <div className="shrink-0 mt-0.5">
              <Ion name="alert-circle" size={24} color="#f87171" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-red-300 font-semibold mb-1 text-[15px]">
                Something went wrong
              </p>
              <p className="text-red-400/80 text-sm leading-relaxed break-words">
                {errorMsg}
              </p>
              <button
                onClick={() => window.location.reload()}
                className="mt-3 inline-flex items-center gap-1.5 text-sm text-red-300 hover:text-white underline underline-offset-2 transition-colors"
              >
                <Ion name="refresh-outline" size={14} color="currentColor" />
                Reload and try again
              </button>
            </div>
          </div>
        )}

        {/* ────────────────────────────────────────────────────────────────
            VIEW A: Upload screen (no image selected yet)
        ──────────────────────────────────────────────────────────────── */}
        {!originalSrc && status !== "error" && (
          <>
            {/* Hero headline */}
            <div className="text-center mb-10">
              <div className="inline-flex items-center gap-2 bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-sm font-medium rounded-full px-4 py-1.5 mb-6">
                <Ion name="sparkles-outline" size={14} color="#a5b4fc" />
                Powered by briaai/RMBG-1.4 · Runs entirely in your browser
              </div>
              <h2 className="text-4xl sm:text-5xl font-extrabold text-white mb-4 leading-[1.1] tracking-tight">
                Remove Image Backgrounds
                <br />
                <span className="bg-gradient-to-r from-indigo-400 via-violet-400 to-purple-500 bg-clip-text text-transparent">
                  Instantly &amp; For Free
                </span>
              </h2>
              <p className="text-slate-400 text-lg max-w-2xl mx-auto leading-relaxed">
                Upload any image and our on-device AI will cleanly separate the
                subject from its background in seconds — no account, no API key,
                no cost.
              </p>
            </div>

            {/* Drag-and-drop upload zone */}
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={openFilePicker}
              role="button"
              tabIndex={isReady ? 0 : -1}
              onKeyDown={(e) => e.key === "Enter" && openFilePicker()}
              aria-label="Upload image"
              className={`
                relative border-2 border-dashed rounded-3xl p-12 sm:p-20 text-center
                transition-all duration-300 group outline-none
                ${
                  !isReady
                    ? "border-slate-700 opacity-60 cursor-not-allowed"
                    : isDragOver
                    ? "border-indigo-400 bg-indigo-500/10 scale-[1.01] cursor-copy shadow-xl shadow-indigo-500/10"
                    : "border-slate-700 hover:border-indigo-500/50 hover:bg-slate-800/30 cursor-pointer"
                }
              `}
            >
              {/* Corner decorations */}
              {["top-4 left-4 border-t-2 border-l-2 rounded-tl", "top-4 right-4 border-t-2 border-r-2 rounded-tr", "bottom-4 left-4 border-b-2 border-l-2 rounded-bl", "bottom-4 right-4 border-b-2 border-r-2 rounded-br"].map(
                (cls, i) => (
                  <div
                    key={i}
                    className={`absolute w-4 h-4 border-slate-600 ${cls} opacity-40`}
                  />
                )
              )}

              {/* Upload icon */}
              <div
                className={`mx-auto w-24 h-24 rounded-2xl flex items-center justify-center mb-6 transition-all duration-300 ${
                  isDragOver
                    ? "bg-indigo-500/20 scale-110"
                    : "bg-slate-800/80 group-hover:bg-slate-800"
                }`}
              >
                <Ion
                  name={isDragOver ? "cloud-download-outline" : "cloud-upload-outline"}
                  size={44}
                  color={isDragOver ? "#818cf8" : "#475569"}
                />
              </div>

              {isModelLoading ? (
                <>
                  <p className="text-slate-300 text-xl font-semibold mb-2">
                    AI Model Loading…
                  </p>
                  <p className="text-slate-500 text-sm">
                    Please wait while the model downloads (this happens once).
                  </p>
                </>
              ) : isDragOver ? (
                <p className="text-indigo-300 text-xl font-semibold">
                  Release to remove background!
                </p>
              ) : (
                <>
                  <p className="text-white text-xl font-semibold mb-2">
                    Drag &amp; drop your image here
                  </p>
                  <p className="text-slate-500 text-sm mb-7">
                    Supports JPG, PNG, WEBP, GIF, BMP and more
                  </p>
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); openFilePicker(); }}
                    disabled={!isReady}
                    className="inline-flex items-center gap-2.5 bg-indigo-600 hover:bg-indigo-500 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold px-7 py-3.5 rounded-2xl transition-all duration-200 shadow-xl shadow-indigo-500/25 hover:shadow-indigo-500/40 text-[15px]"
                  >
                    <Ion name="folder-open-outline" size={18} color="white" />
                    Choose Image
                  </button>
                </>
              )}
            </div>

            {/* Feature cards */}
            <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4">
              {[
                {
                  icon: "shield-checkmark-outline",
                  color: "#34d399",
                  bg: "from-emerald-500/10 to-emerald-500/5",
                  border: "border-emerald-500/20",
                  title: "100% Private",
                  desc: "Your images never leave your device. All ML inference runs locally via WebAssembly.",
                },
                {
                  icon: "flash-outline",
                  color: "#fbbf24",
                  bg: "from-amber-500/10 to-amber-500/5",
                  border: "border-amber-500/20",
                  title: "State-of-the-Art AI",
                  desc: "Uses RMBG-1.4 by BRIA AI — one of the best open-source background removal models.",
                },
                {
                  icon: "infinite-outline",
                  color: "#818cf8",
                  bg: "from-indigo-500/10 to-indigo-500/5",
                  border: "border-indigo-500/20",
                  title: "Unlimited & Free",
                  desc: "No file size limits, no watermarks, no sign-ups. Process as many images as you want.",
                },
              ].map(({ icon, color, bg, border, title, desc }) => (
                <div
                  key={title}
                  className={`bg-gradient-to-br ${bg} border ${border} rounded-2xl p-5 flex gap-4`}
                >
                  <div className="shrink-0 w-10 h-10 rounded-xl bg-slate-800/60 flex items-center justify-center">
                    <Ion name={icon} size={20} color={color} />
                  </div>
                  <div>
                    <p className="text-white font-semibold text-sm mb-1">{title}</p>
                    <p className="text-slate-400 text-xs leading-relaxed">{desc}</p>
                  </div>
                </div>
              ))}
            </div>

            {/* How it works */}
            <div className="mt-10 text-center">
              <p className="text-slate-600 text-xs uppercase tracking-widest mb-5 font-medium">
                How It Works
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-0">
                {[
                  { step: "1", icon: "cloud-upload-outline", label: "Upload Image" },
                  { step: "2", icon: "cpu-outline", label: "AI Processes Locally" },
                  { step: "3", icon: "download-outline", label: "Download PNG" },
                ].map(({ step, icon, label }, i) => (
                  <div key={step} className="flex items-center">
                    <div className="flex flex-col items-center gap-2">
                      <div className="w-12 h-12 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center relative">
                        <Ion name={icon} size={22} color="#64748b" />
                        <span className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-indigo-600 text-white text-[10px] font-bold flex items-center justify-center">
                          {step}
                        </span>
                      </div>
                      <p className="text-slate-400 text-xs font-medium">{label}</p>
                    </div>
                    {i < 2 && (
                      <div className="hidden sm:block mx-4 text-slate-700">
                        <Ion name="chevron-forward-outline" size={16} color="#334155" />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {/* ────────────────────────────────────────────────────────────────
            VIEW B: Comparison view (image uploaded)
        ──────────────────────────────────────────────────────────────── */}
        {originalSrc && status !== "error" && (
          <div className="space-y-5">
            {/* ── Toolbar ─────────────────────────────────────────────── */}
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div className="flex items-center gap-2 min-w-0">
                <Ion name="images-outline" size={18} color="#94a3b8" />
                <span className="text-white font-semibold text-sm truncate max-w-[160px] sm:max-w-xs">
                  {fileName || "Image"}
                </span>
                {isProcessing && (
                  <span className="flex items-center gap-1.5 text-indigo-400 text-xs ml-1">
                    <Spinner size={11} color="#818cf8" />
                    {statusText}
                  </span>
                )}
                {isDone && (
                  <span className="flex items-center gap-1 text-emerald-400 text-xs ml-1">
                    <Ion name="checkmark-circle" size={13} color="#34d399" />
                    Background removed!
                  </span>
                )}
              </div>

              <div className="flex items-center gap-2 shrink-0">
                {/* New image */}
                <button
                  type="button"
                  onClick={handleReset}
                  className="flex items-center gap-1.5 text-slate-400 hover:text-white border border-slate-700 hover:border-slate-500 bg-slate-800/50 hover:bg-slate-800 px-3 py-2 rounded-xl text-sm font-medium transition-all duration-200"
                >
                  <Ion name="arrow-back-outline" size={14} color="currentColor" />
                  New Image
                </button>

                {/* Download PNG — only visible when processing is done */}
                {isDone && (
                  <button
                    type="button"
                    onClick={handleDownload}
                    className="flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-semibold px-4 py-2 rounded-xl text-sm transition-all duration-200 shadow-lg shadow-indigo-500/30 hover:shadow-indigo-500/50 hover:-translate-y-0.5 active:translate-y-0"
                  >
                    <Ion name="download-outline" size={16} color="white" />
                    Download PNG
                  </button>
                )}
              </div>
            </div>

            {/* ── Side-by-side panels ─────────────────────────────────── */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Original panel */}
              <div className="bg-slate-900/60 border border-slate-700/50 rounded-2xl overflow-hidden flex flex-col">
                <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-700/40 bg-slate-800/40 shrink-0">
                  <Ion name="image-outline" size={15} color="#64748b" />
                  <span className="text-slate-400 text-sm font-medium">Original</span>
                </div>
                <div className="flex-1 flex items-center justify-center p-4 min-h-56 bg-slate-900/30">
                  <img
                    src={originalSrc}
                    alt="Original"
                    className="max-w-full max-h-96 object-contain rounded-xl shadow-2xl"
                  />
                </div>
              </div>

              {/* Result panel */}
              <div className="bg-slate-900/60 border border-slate-700/50 rounded-2xl overflow-hidden flex flex-col">
                <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700/40 bg-slate-800/40 shrink-0">
                  <div className="flex items-center gap-2">
                    <Ion name="color-wand-outline" size={15} color="#818cf8" />
                    <span className="text-slate-400 text-sm font-medium">Result</span>
                  </div>
                  {isDone && (
                    <div className="flex items-center gap-1 text-slate-500 text-[11px]">
                      <Ion name="grid-outline" size={11} color="#475569" />
                      Transparent PNG
                    </div>
                  )}
                </div>

                {/* Checkerboard background for transparency indication */}
                <div
                  className="flex-1 flex items-center justify-center p-4 min-h-56 relative rounded-b-2xl"
                  style={isDone ? { background: CHECKER_BG } : undefined}
                >
                  {/* Processing spinner */}
                  {isProcessing && (
                    <div className="flex flex-col items-center gap-4 text-center">
                      <div className="relative w-16 h-16 flex items-center justify-center">
                        <div className="absolute inset-0 rounded-full bg-indigo-500/10 animate-pulse" />
                        <Spinner size={32} color="#818cf8" />
                      </div>
                      <div>
                        <p className="text-slate-300 text-sm font-medium">{statusText}</p>
                        <p className="text-slate-500 text-xs mt-0.5">
                          Analysing pixels…
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Result image */}
                  {isDone && resultSrc && (
                    <img
                      src={resultSrc}
                      alt="Background removed"
                      className="max-w-full max-h-96 object-contain rounded-xl drop-shadow-2xl"
                    />
                  )}

                  {/* Placeholder */}
                  {!isProcessing && !isDone && (
                    <div className="text-slate-700 flex flex-col items-center gap-3">
                      <Ion name="scan-outline" size={48} color="#334155" />
                      <p className="text-sm">Result will appear here</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* ── Drop zone for next image ─────────────────────────────── */}
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={openFilePicker}
              role="button"
              tabIndex={isReady ? 0 : -1}
              onKeyDown={(e) => e.key === "Enter" && openFilePicker()}
              className={`
                border border-dashed rounded-2xl py-4 px-6 flex items-center justify-center gap-3
                text-sm transition-all duration-200 cursor-pointer outline-none
                ${
                  !isReady
                    ? "border-slate-800 text-slate-700 cursor-not-allowed"
                    : isDragOver
                    ? "border-indigo-500 bg-indigo-500/10 text-indigo-300"
                    : "border-slate-700 hover:border-slate-600 text-slate-500 hover:text-slate-300 hover:bg-slate-800/30"
                }
              `}
            >
              <Ion name="add-circle-outline" size={18} color="currentColor" />
              <span>
                {isDragOver
                  ? "Drop to process new image"
                  : "Try another image — drag & drop or click here"}
              </span>
            </div>
          </div>
        )}
      </main>

      {/* ══════════════════════════════════════════════════════════════════
          FOOTER
      ══════════════════════════════════════════════════════════════════ */}
      <footer className="border-t border-white/5 bg-slate-950/60 mt-auto">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-5">
            {/* Privacy statement */}
            <div className="flex items-start sm:items-center gap-3">
              <div className="shrink-0 mt-0.5 sm:mt-0 w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                <Ion name="shield-checkmark-outline" size={16} color="#34d399" />
              </div>
              <p className="text-slate-500 text-xs sm:text-sm leading-relaxed">
                <strong className="text-slate-400 font-semibold">Privacy First:</strong>{" "}
                All AI processing happens 100% locally on your device using WebAssembly.{" "}
                <strong className="text-slate-400">No images are ever uploaded to any server.</strong>
              </p>
            </div>

            {/* Tech attribution */}
            <div className="flex items-center gap-3 text-slate-600 text-xs shrink-0">
              <Ion name="cpu-outline" size={13} color="#475569" />
              <span>Powered by</span>
              <a
                href="https://huggingface.co/briaai/RMBG-1.4"
                target="_blank"
                rel="noopener noreferrer"
                className="text-indigo-500 hover:text-indigo-400 underline underline-offset-2 transition-colors"
              >
                RMBG-1.4
              </a>
              <span className="text-slate-700">·</span>
              <a
                href="https://github.com/xenova/transformers.js"
                target="_blank"
                rel="noopener noreferrer"
                className="text-indigo-500 hover:text-indigo-400 underline underline-offset-2 transition-colors"
              >
                Transformers.js
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
