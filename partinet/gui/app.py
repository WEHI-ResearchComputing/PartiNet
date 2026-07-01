import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from partinet.gui.job_registry import cancel_job, jobs_dropdown_update, jobs_markdown, read_job_log
from partinet.gui.job_runner import JobSpec, SlurmOptions, slurm_field_defaults, stream_job
from partinet.process_utils.image_io import is_micrograph_file, micrograph_dimensions, load_micrograph_for_detect

# ── Branding ──────────────────────────────────────────────────────────────────

_THEME_NAME = "lone17/kotaemon"

_DEFAULT_DARK_HEAD = """
<script>
(function () {
  var url = new URL(window.location.href);
  if (url.searchParams.get("__theme") !== "dark") {
    url.searchParams.set("__theme", "dark");
    window.location.replace(url.toString());
  }
})();
</script>
"""


def _resolve_theme():
    try:
        return gr.themes.Base.from_hub(_THEME_NAME)
    except Exception:
        return gr.themes.Default()


_THEME = _resolve_theme()

_HEADER_HTML = """
<div style="padding:8px 0 4px 0">
  <div style="font-size:1.6rem;font-weight:700;line-height:1.1">PartiNet</div>
  <div style="font-size:0.85rem;opacity:0.7">
    Automated cryo-EM particle picker &nbsp;·&nbsp;
    Run each stage in order: <b>1&nbsp;·&nbsp;Denoise → 2&nbsp;·&nbsp;Detect → 3&nbsp;·&nbsp;Star&nbsp;File</b>
  </div>
</div>
"""


def _env_default(name: str) -> str:
    return os.environ.get(name, "").strip()


def _slurm_from_ui(mode, partition, account, time_limit, cpus, gpus, mem, extra, preamble, partinet_cmd):
    return SlurmOptions(
        partition=(partition or "").strip(),
        account=(account or "").strip(),
        time_limit=(time_limit or "").strip(),
        cpus=str(cpus).strip() if cpus not in (None, "") else "",
        gpus=str(gpus).strip() if gpus not in (None, "") else "",
        mem=(mem or "").strip(),
        extra_sbatch=(extra or "").strip(),
        preamble=(preamble or "").strip(),
        partinet_cmd=(partinet_cmd or "partinet").strip() or "partinet",
    ), _normalize_mode(mode)


def _normalize_mode(mode: str) -> str:
    m = (mode or "Local").strip().lower()
    return "slurm" if m == "slurm" else "local"


def _project_log(project: str, name: str) -> str:
    return os.path.join(project.strip(), name)


def _run_job(spec: JobSpec):
    yield from stream_job(spec)


# ── Stage runners ────────────────────────────────────────────────────────────

def run_denoise(source, project, img_format, num_workers, mode, partition, account, time_limit, cpus, gpus, mem, extra, preamble, partinet_cmd):
    source, project = source.strip(), project.strip()
    if not source:
        yield "ERROR: Raw micrographs directory is required."
        return
    if not project:
        yield "ERROR: Project directory is required."
        return

    os.makedirs(project, exist_ok=True)
    cmd = ["partinet", "denoise", "--source", source, "--project", project, "--img_format", img_format]
    if num_workers not in (None, ""):
        cmd.extend(["--num_workers", str(int(num_workers))])

    slurm, job_mode = _slurm_from_ui(mode, partition, account, time_limit, cpus, gpus, mem, extra, preamble, partinet_cmd)
    spec = JobSpec(
        stage="denoise",
        command=cmd,
        project_dir=project,
        log_path=_project_log(project, "partinet_denoise.log"),
        mode=job_mode,
        slurm=slurm,
    )
    yield from _run_job(spec)


def run_detect(weight, source, project, conf_thres, iou_thres, device, img_size, dy_thres, mode, partition, account, time_limit, cpus, gpus, mem, extra, preamble, partinet_cmd):
    weight, source, project = weight.strip(), source.strip(), project.strip()
    if not weight:
        yield "ERROR: Model weights path is required."
        return
    if not source:
        yield "ERROR: Source images directory is required."
        return
    if not project:
        yield "ERROR: Project directory is required."
        return

    cmd = [
        "partinet", "detect",
        "--weight", weight,
        "--source", source,
        "--project", project,
        "--conf-thres", str(conf_thres),
        "--iou-thres", str(iou_thres),
        "--img-size", str(int(img_size)),
        "--dy-thres", str(dy_thres),
        "--exist-ok",
    ]
    if device.strip():
        cmd.extend(["--device", device.strip()])

    slurm, job_mode = _slurm_from_ui(mode, partition, account, time_limit, cpus, gpus, mem, extra, preamble, partinet_cmd)
    spec = JobSpec(
        stage="detect",
        command=cmd,
        project_dir=project,
        log_path=_project_log(project, "partinet_detect.log"),
        mode=job_mode,
        slurm=slurm,
    )
    yield from _run_job(spec)


def run_star(labels, images, output, conf, relion, relion_project_dir, mrc_prefix, mode, partition, account, time_limit, cpus, gpus, mem, extra, preamble, partinet_cmd):
    labels, images, output = labels.strip(), images.strip(), output.strip()
    relion_project_dir = (relion_project_dir or "").strip()
    mrc_prefix = (mrc_prefix or "").strip()

    if not labels:
        yield "ERROR: Labels directory is required."
        return
    if not images:
        yield "ERROR: Images directory is required."
        return
    if not output:
        yield "ERROR: Output STAR file path is required."
        return
    if relion and not relion_project_dir:
        yield "ERROR: RELION project directory is required when RELION output is enabled."
        return

    project = os.path.dirname(os.path.abspath(output))
    cmd = [
        "partinet", "star",
        "--labels", labels,
        "--images", images,
        "--output", output,
        "--conf", str(conf),
    ]
    if relion:
        cmd.extend(["--relion", "--relion-project-dir", relion_project_dir])
        if mrc_prefix:
            cmd.extend(["--mrc-prefix", mrc_prefix])

    slurm, job_mode = _slurm_from_ui(mode, partition, account, time_limit, cpus, gpus, mem, extra, preamble, partinet_cmd)
    spec = JobSpec(
        stage="star",
        command=cmd,
        project_dir=project,
        log_path=_project_log(project, "partinet_star.log"),
        mode=job_mode,
        slurm=slurm,
    )
    yield from _run_job(spec)


# ── Star File analysis ───────────────────────────────────────────────────────

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".mrc")


def _parse_label_file(path):
    dets = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                dets.append({
                    "x": float(parts[1]), "y": float(parts[2]),
                    "w": float(parts[3]), "h": float(parts[4]),
                    "conf": float(parts[5]) if len(parts) >= 6 else 1.0,
                })
    return dets


def _img_size(path):
    try:
        return micrograph_dimensions(path)
    except (ValueError, OSError):
        return 4096, 4096


def load_detections(labels_dir, images_dir):
    labels_dir, images_dir = labels_dir.strip(), images_dir.strip()
    _err = lambda msg: (None, msg, None, None, None, None, gr.update(choices=[]), "")

    if not labels_dir or not images_dir:
        return _err("Both directories are required.")
    if not os.path.isdir(labels_dir):
        return _err(f"Labels directory not found: `{labels_dir}`")
    if not os.path.isdir(images_dir):
        return _err(f"Images directory not found: `{images_dir}`")

    label_files = sorted(f for f in os.listdir(labels_dir) if f.endswith(".txt"))
    if not label_files:
        return _err("No .txt label files found in labels directory.")

    mics, all_confs, all_sizes = [], [], []
    for lf in label_files:
        stem = os.path.splitext(lf)[0]
        img_path = next(
            (os.path.join(images_dir, stem + ext) for ext in _IMG_EXTS
             if os.path.exists(os.path.join(images_dir, stem + ext))),
            None,
        )
        if img_path is None:
            continue
        w, h = _img_size(img_path)
        dets = _parse_label_file(os.path.join(labels_dir, lf))
        for d in dets:
            all_confs.append(d["conf"])
            all_sizes.append(max(d["w"] * w, d["h"] * h))
        mics.append({"name": stem, "img": img_path, "w": w, "h": h, "dets": dets})

    if not mics:
        return (None, "No label files had matching images in the images directory.", None, None, None, None, gr.update(choices=[]), "")

    state = {"mics": mics, "confs": all_confs, "sizes": all_sizes}
    n, m = len(all_confs), len(mics)
    counts = sorted(len(mic["dets"]) for mic in mics)
    summary = (
        f"**{n:,} detections across {m:,} micrographs** — "
        f"mean {n/m:.0f} · median {counts[m//2]} · "
        f"range {counts[0]}–{counts[-1]} particles/micrograph"
    )

    DEFAULT = 0.1
    choices = [mic["name"] for mic in mics]
    return (
        state, summary,
        _conf_plot(all_confs, DEFAULT),
        _size_plot(all_sizes),
        _mic_count_plot(mics, DEFAULT),
        _bivariate_plot(all_confs, all_sizes),
        gr.update(choices=choices, value=choices[0]),
        _retained_text(all_confs, DEFAULT),
    )


def _conf_plot(confs, threshold):
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(confs, bins=50, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.axvline(threshold, color="crimson", linestyle="--", linewidth=1.5, label=f"Threshold {threshold:.2f}")
        ax.legend(fontsize=8)
        ax.set_xlabel("Confidence score")
        ax.set_ylabel("Detections")
        ax.set_title("Confidence distribution")
        fig.tight_layout()
    return fig


def _size_plot(sizes_px):
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(sizes_px, bins=50, color="darkorange", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Box size (px)")
        ax.set_ylabel("Detections")
        ax.set_title("Box size distribution")
        fig.tight_layout()
    return fig


def _mic_count_plot(mics, threshold):
    with plt.style.context("dark_background"):
        counts = [sum(1 for d in m["dets"] if d["conf"] >= threshold) for m in mics]
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.bar(range(len(counts)), counts, color="mediumseagreen", width=1.0, linewidth=0)
        ax.set_xlabel("Micrograph index")
        ax.set_ylabel("Particles")
        ax.set_title(f"Particles per micrograph  (threshold = {threshold:.2f})")
        fig.tight_layout()
    return fig


def _bivariate_plot(confs, sizes):
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(5, 3))
        h = ax.hist2d(confs, sizes, bins=50, cmap="viridis")
        fig.colorbar(h[3], ax=ax, label="Detections")
        ax.set_xlabel("Confidence score")
        ax.set_ylabel("Box size (px)")
        ax.set_title("Confidence vs box size")
        fig.tight_layout()
    return fig


def _retained_text(confs, threshold):
    total = len(confs)
    kept = sum(1 for c in confs if c >= threshold)
    pct = 100 * kept / total if total > 0 else 0
    return f"**{kept:,} / {total:,} particles retained** ({pct:.1f}%)"


def _draw_detections(mic, threshold, max_px=1000):
    from PIL import Image as _PIL, ImageDraw
    import cv2
    if mic["img"].lower().endswith(".mrc"):
        bgr = load_micrograph_for_detect(mic["img"])
        img = _PIL.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    else:
        img = _PIL.open(mic["img"]).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_px / max(w, h))
    dw, dh = int(w * scale), int(h * scale)
    if scale < 1.0:
        img = img.resize((dw, dh), _PIL.LANCZOS)
    draw = ImageDraw.Draw(img)
    n_shown = 0
    for d in mic["dets"]:
        if d["conf"] < threshold:
            continue
        cx, cy = d["x"] * dw, d["y"] * dh
        bw, bh = d["w"] * dw, d["h"] * dh
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        c = d["conf"]
        color = (int((1 - c) * 220), int(c * 220), 60)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        n_shown += 1
    return img, n_shown


def update_threshold(state, threshold, mic_name):
    if state is None:
        return "", None, None, None, ""
    retained = _retained_text(state["confs"], threshold)
    conf_fig = _conf_plot(state["confs"], threshold)
    mic_count_fig = _mic_count_plot(state["mics"], threshold)
    img, mic_stats = None, ""
    if mic_name:
        mic = next((m for m in state["mics"] if m["name"] == mic_name), None)
        if mic:
            img, n = _draw_detections(mic, threshold)
            mic_stats = f"**{n}** particles shown"
    return retained, conf_fig, mic_count_fig, img, mic_stats


def update_micrograph(state, mic_name, threshold):
    if state is None or not mic_name:
        return None, ""
    mic = next((m for m in state["mics"] if m["name"] == mic_name), None)
    if mic is None:
        return None, ""
    img, n = _draw_detections(mic, threshold)
    return img, f"**{n}** particles shown"


# ── Helpers for global project directory ─────────────────────────────────────

def _find_labels_dirs(project_dir):
    import glob as _glob
    p = (project_dir or "").strip()
    if not p or not os.path.isdir(p):
        return []
    exp_dirs = sorted(
        [d for d in _glob.glob(os.path.join(p, "exp*")) if os.path.isdir(d)],
        key=os.path.getmtime,
        reverse=True,
    )
    return [
        os.path.join(d, "labels")
        for d in exp_dirs
        if os.path.isdir(os.path.join(d, "labels"))
    ]


def refresh_labels(project_dir):
    dirs = _find_labels_dirs(project_dir)
    return dirs[0] if dirs else ""


def update_project_dir(project_dir):
    p = (project_dir or "").strip()
    if not p:
        return ("", "", "", "", "", "", jobs_markdown(""), jobs_dropdown_update(""), "")
    denoised = os.path.join(p, "denoised")
    dirs = _find_labels_dirs(p)
    labels_val = dirs[0] if dirs else os.path.join(p, "exp", "labels")
    return (
        p,                                          # d1_project
        denoised,                                   # d2_source
        p,                                          # d2_project
        labels_val,                                 # d3_labels
        denoised,                                   # d3_images
        os.path.join(p, "particles.star"),          # d3_output
        jobs_markdown(p),
        jobs_dropdown_update(p),
        "",
    )


def refresh_jobs(project_dir, job_key=None):
    log = read_job_log(project_dir, job_key or "")
    return jobs_markdown(project_dir), jobs_dropdown_update(project_dir), log


def view_job_log(project_dir, job_key):
    return read_job_log(project_dir, job_key or "")


def cancel_selected_job(project_dir, job_key):
    message = cancel_job(job_key or "", project_dir)
    markdown, dropdown, log = refresh_jobs(project_dir, job_key)
    return markdown, dropdown, log, message


# ── Gradio UI ────────────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="PartiNet") as app:
        gr.HTML(_HEADER_HTML)

        gr_project = gr.Textbox(
            label="Project directory",
            value=_env_default("PARTINET_PROJECT"),
            placeholder="/path/to/my_project",
            info="Set once — auto-fills project paths in all three stages below",
        )

        with gr.Accordion("Running jobs", open=False):
            jobs_markdown_out = gr.Markdown("Set a project directory to view jobs.")
            job_select = gr.Dropdown(
                label="Select job",
                choices=[],
                interactive=True,
                info="Select a running job to view its log below",
            )
            job_log_view = gr.Textbox(
                label="Job log",
                lines=16,
                max_lines=40,
                interactive=False,
            )
            with gr.Row():
                jobs_refresh_btn = gr.Button("Refresh", variant="secondary")
                jobs_cancel_btn = gr.Button("Cancel selected", variant="stop")
            jobs_cancel_status = gr.Markdown("")
            jobs_timer = gr.Timer(5)

        with gr.Accordion("Execution settings (Local / Slurm)", open=False):
            gr.Markdown(
                "Run stages on this machine (**Local**) or submit batch jobs (**Slurm**). "
                "Resource defaults update per stage when you switch tabs "
                "(denoise/detect: 32 CPUs, 100G RAM; detect adds 4 GPUs; star: 16 CPUs, 64G RAM). "
                "Leave fields blank to use those defaults. "
                "Optional cluster-wide overrides: set `PARTINET_SLURM_CONFIG` to a YAML path."
            )
            exec_mode = gr.Radio(["Local", "Slurm"], value="Local", label="Execution mode")
            with gr.Row():
                slurm_partition = gr.Textbox(label="Partition", placeholder="")
                slurm_account = gr.Textbox(label="Account", placeholder="")
                slurm_time = gr.Textbox(label="Time limit", placeholder="HH:MM:SS")
            with gr.Row():
                slurm_cpus = gr.Textbox(label="CPUs per task", value="32", placeholder="32")
                slurm_gpus = gr.Textbox(label="GPUs (gres count)", value="", placeholder="")
                slurm_mem = gr.Textbox(label="Memory", value="100G", placeholder="100G")
            slurm_extra = gr.Textbox(
                label="Extra #SBATCH lines",
                placeholder="#SBATCH --constraint=...",
                lines=2,
            )
            slurm_preamble = gr.Textbox(
                label="Job setup script",
                placeholder="# module load ...\\n# source activate ...",
                lines=3,
            )
            slurm_partinet_cmd = gr.Textbox(
                label="PartiNet executable",
                value="partinet",
                placeholder="partinet",
            )

        slurm_inputs = [
            exec_mode, slurm_partition, slurm_account, slurm_time,
            slurm_cpus, slurm_gpus, slurm_mem, slurm_extra, slurm_preamble, slurm_partinet_cmd,
        ]

        with gr.Tabs():

            # ── 1. Denoise ───────────────────────────────────────────────────
            with gr.Tab("1 · Denoise") as tab_denoise:
                gr.Markdown(
                    "Improve signal-to-noise in raw micrographs using a Wiener filter. "
                    "Output images are saved to `project/denoised/`."
                )
                with gr.Row():
                    d1_source = gr.Textbox(
                        label="Raw micrographs directory",
                        placeholder="/path/to/motion_corrected",
                        value="",
                        info="Folder of .mrc files from RELION or CryoSPARC motion correction",
                    )
                    d1_project = gr.Textbox(
                        label="Project directory",
                        placeholder="/path/to/my_project",
                        value=_env_default("PARTINET_PROJECT"),
                        info="All PartiNet outputs for this dataset will be written here",
                    )
                with gr.Row():
                    d1_fmt = gr.Dropdown(
                        choices=["png", "jpg", "mrc"],
                        value="png",
                        label="Output image format",
                        info="PNG recommended — lossless and directly compatible with Detect",
                    )
                    d1_workers = gr.Number(
                        label="CPU workers (blank = auto)",
                        value=None,
                        precision=0,
                        minimum=1,
                        info="Parallel workers for denoising; auto uses half the available CPUs",
                    )
                d1_btn = gr.Button("▶  Run Denoise", variant="primary")
                d1_log = gr.Textbox(
                    label="Log output",
                    lines=14,
                    max_lines=30,
                    interactive=False,
                )
                d1_evt = d1_btn.click(
                    run_denoise,
                    inputs=[d1_source, d1_project, d1_fmt, d1_workers] + slurm_inputs,
                    outputs=d1_log,
                )

            # ── 2. Detect ────────────────────────────────────────────────────
            with gr.Tab("2 · Detect") as tab_detect:
                gr.Markdown(
                    "Locate particles in denoised micrographs using the DynamicDet model. "
                    "Results are saved to `project/exp/`."
                )
                with gr.Row():
                    d2_weight = gr.Textbox(
                        label="Model weights (.pt)",
                        placeholder="/path/to/model.pt",
                        value=_env_default("PARTINET_WEIGHTS"),
                        info="Pre-trained weights — download from HuggingFace or use your own trained model",
                    )
                    d2_source = gr.Textbox(
                        label="Denoised images directory",
                        placeholder="/path/to/my_project/denoised",
                        value="",
                        info="Output of the Denoise step (project/denoised/)",
                    )
                with gr.Row():
                    d2_project = gr.Textbox(
                        label="Project directory",
                        placeholder="/path/to/my_project",
                        value=_env_default("PARTINET_PROJECT"),
                        info="Same project directory used in Denoise",
                    )
                    d2_device = gr.Textbox(
                        label="GPU device(s)",
                        value="",
                        placeholder="0  or  0,1,2,3  or  cpu",
                        info="Leave blank to auto-detect; use 'cpu' if no GPU is available",
                    )
                with gr.Row():
                    d2_conf = gr.Slider(
                        0.0, 1.0, value=0.1, step=0.01,
                        label="Confidence threshold",
                        info="Lower → more picks (including false positives). Recommended: 0.0–0.3",
                    )
                    d2_iou = gr.Slider(
                        0.0, 1.0, value=0.2, step=0.01,
                        label="IOU threshold",
                        info="Controls removal of overlapping detections. Default 0.2 works for most datasets",
                    )
                with gr.Accordion("Advanced options", open=False):
                    with gr.Row():
                        d2_imgsize = gr.Number(
                            label="Inference image size (px)",
                            value=1280,
                            precision=0,
                            info="Must match the size used during training (default: 1280)",
                        )
                        d2_dy = gr.Slider(
                            0.0, 1.0, value=0.5, step=0.01,
                            label="Dynamic threshold",
                            info="Router threshold between easy/hard micrograph detectors",
                        )
                d2_btn = gr.Button("▶  Run Detect", variant="primary")
                d2_log = gr.Textbox(
                    label="Log output",
                    lines=14,
                    max_lines=30,
                    interactive=False,
                )
                d2_evt = d2_btn.click(
                    run_detect,
                    inputs=[
                        d2_weight, d2_source, d2_project, d2_conf, d2_iou, d2_device, d2_imgsize, d2_dy,
                    ] + slurm_inputs,
                    outputs=d2_log,
                )

            # ── 3. Star File ─────────────────────────────────────────────────
            with gr.Tab("3 · Star File") as tab_star:
                gr.Markdown(
                    "Load detections from a Detect run, explore statistics, set a confidence "
                    "threshold interactively, then generate a STAR file for **CryoSPARC** or **RELION**."
                )

                # --- Load ---
                with gr.Row():
                    d3_labels = gr.Textbox(
                        label="Labels directory",
                        placeholder="/path/to/project/exp/labels",
                        value="",
                        info="Folder of .txt detection files from the Detect step",
                    )
                    d3_images = gr.Textbox(
                        label="Denoised images directory",
                        placeholder="/path/to/project/denoised",
                        value="",
                        info="Used to resolve image dimensions for coordinate conversion",
                    )
                d3_state = gr.State(None)
                d3_load_btn = gr.Button("Load Detections", variant="secondary")
                d3_summary = gr.Markdown("")

                # --- Statistics ---
                with gr.Row():
                    d3_conf_plot = gr.Plot(format="png", label="Confidence distribution")
                    d3_size_plot = gr.Plot(format="png", label="Box size distribution")
                with gr.Row():
                    d3_mic_count_plot = gr.Plot(format="png", label="Particles per micrograph")
                    d3_bivariate_plot = gr.Plot(format="png", label="Confidence vs box size")

                # --- Threshold + preview ---
                d3_thresh = gr.Slider(
                    0.0, 1.0, value=0.1, step=0.01,
                    label="Confidence threshold",
                    info="Adjust to filter detections — updates plots and image preview in real time",
                )
                d3_retained = gr.Markdown("")
                d3_mic_select = gr.Dropdown(label="Preview micrograph", choices=[], interactive=True)
                d3_mic_stats = gr.Markdown("")
                d3_preview = gr.Image(label="Detection preview", type="pil")

                # --- STAR generation ---
                gr.Markdown("---\n### Generate STAR File\nThe confidence threshold above is applied.")
                d3_output = gr.Textbox(
                    label="Output STAR file",
                    placeholder="/path/to/project/particles.star",
                    value="",
                    info="CryoSPARC-compatible STAR file will be written here",
                )
                with gr.Accordion("RELION output (optional)", open=False):
                    d3_relion = gr.Checkbox(label="Also generate RELION-format output", value=False)
                    d3_relion_dir = gr.Textbox(
                        label="RELION project directory",
                        placeholder="/path/to/relion_project",
                        info="Creates <project>/partinet/pick.star and per-micrograph coordinate files",
                    )
                    d3_mrc_prefix = gr.Textbox(
                        label="MRC path prefix",
                        placeholder="MotionCorr/job003/movies",
                        value="",
                        info="Prepended to micrograph names in the RELION STAR file",
                    )
                d3_star_btn = gr.Button("▶  Generate STAR File", variant="primary")
                d3_log = gr.Textbox(
                    label="Log output",
                    lines=8,
                    max_lines=20,
                    interactive=False,
                )

                # --- Events ---
                d3_load_btn.click(
                    load_detections,
                    inputs=[d3_labels, d3_images],
                    outputs=[d3_state, d3_summary, d3_conf_plot, d3_size_plot, d3_mic_count_plot, d3_bivariate_plot, d3_mic_select, d3_retained],
                )
                d3_thresh.change(
                    update_threshold,
                    inputs=[d3_state, d3_thresh, d3_mic_select],
                    outputs=[d3_retained, d3_conf_plot, d3_mic_count_plot, d3_preview, d3_mic_stats],
                )
                d3_mic_select.change(
                    update_micrograph,
                    inputs=[d3_state, d3_mic_select, d3_thresh],
                    outputs=[d3_preview, d3_mic_stats],
                )
                d3_evt = d3_star_btn.click(
                    run_star,
                    inputs=[
                        d3_labels, d3_images, d3_output, d3_thresh, d3_relion, d3_relion_dir, d3_mrc_prefix,
                    ] + slurm_inputs,
                    outputs=d3_log,
                )

        jobs_outputs = [jobs_markdown_out, job_select, job_log_view]
        jobs_refresh_btn.click(refresh_jobs, inputs=[gr_project, job_select], outputs=jobs_outputs)
        jobs_timer.tick(refresh_jobs, inputs=[gr_project, job_select], outputs=jobs_outputs)
        job_select.change(view_job_log, inputs=[gr_project, job_select], outputs=job_log_view)
        jobs_cancel_btn.click(
            cancel_selected_job,
            inputs=[gr_project, job_select],
            outputs=[jobs_markdown_out, job_select, job_log_view, jobs_cancel_status],
            cancels=[d1_evt, d2_evt, d3_evt],
        )

        gr_project.change(
            update_project_dir,
            inputs=[gr_project],
            outputs=[
                d1_project, d2_source, d2_project, d3_labels, d3_images, d3_output,
                jobs_markdown_out, job_select, job_log_view,
            ],
        )

        slurm_resource_outputs = [slurm_cpus, slurm_gpus, slurm_mem]
        tab_denoise.select(lambda: slurm_field_defaults("denoise"), outputs=slurm_resource_outputs)
        tab_detect.select(lambda: slurm_field_defaults("detect"), outputs=slurm_resource_outputs)
        tab_star.select(lambda: slurm_field_defaults("star"), outputs=slurm_resource_outputs)

    return app


def launch_gui(host="127.0.0.1", port=None, share=False):
    import click

    app = build_app()
    app.queue()
    browse_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    if port is not None:
        click.echo(f"PartiNet GUI — open http://{browse_host}:{port} on this machine.")
        click.echo("Over SSH, forward the port from your laptop, then browse to localhost:")
        click.echo(f"  ssh -L {port}:127.0.0.1:{port} user@login-node")
        click.echo(f"  http://localhost:{port}")
    else:
        click.echo(f"PartiNet GUI — open http://{browse_host}:<port> on this machine.")
        click.echo(
            "Over SSH, forward the port Gradio prints below, then open http://localhost:<port> "
            "in your local browser."
        )
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=_THEME,
        head=_DEFAULT_DARK_HEAD,
        ssr_mode=False,
    )
