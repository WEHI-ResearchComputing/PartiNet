import os
import queue
import logging
import threading
import traceback
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr


# ── Log capture ──────────────────────────────────────────────────────────────

class _QueueHandler(logging.Handler):
    def __init__(self, q):
        super().__init__()
        self.q = q

    def emit(self, record):
        self.q.put(self.format(record) + "\n")


def _stream_stage(fn, args, loggers):
    """Run fn(*args) in a daemon thread; yield accumulated log text for Gradio streaming."""
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    log_q = queue.Queue()
    handler = _QueueHandler(log_q)
    handler.setFormatter(fmt)

    for name in loggers:
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        if lg.level == logging.NOTSET or lg.level > logging.INFO:
            lg.setLevel(logging.INFO)

    result = {"done": False, "error": None}

    def _target():
        try:
            fn(*args)
        except Exception:
            result["error"] = traceback.format_exc()
        finally:
            result["done"] = True

    threading.Thread(target=_target, daemon=True).start()

    output = ""
    while not result["done"]:
        time.sleep(1.0)
        while not log_q.empty():
            output += log_q.get_nowait()
        yield output

    # drain any messages that arrived after the thread set done=True
    while not log_q.empty():
        output += log_q.get_nowait()

    for name in loggers:
        logging.getLogger(name).removeHandler(handler)

    if result["error"]:
        output += f"\n--- FAILED ---\n{result['error']}"
    else:
        output += "\n--- Complete ---"

    yield output


# ── Stage runners ────────────────────────────────────────────────────────────

def run_denoise(source, project, img_format, num_workers):
    if not source.strip():
        yield "ERROR: Raw micrographs directory is required."
        return
    if not project.strip():
        yield "ERROR: Project directory is required."
        return

    ncpu = int(num_workers) if num_workers else None
    os.makedirs(project.strip(), exist_ok=True)
    log_path = os.path.join(project.strip(), "partinet_denoise.log")

    import partinet.process_utils.pooled_denoise_proc as _m

    result = {"done": False, "error": None}

    def _target():
        try:
            _m.main(source.strip(), project.strip(), ncpu, img_format)
        except Exception:
            result["error"] = traceback.format_exc()
        finally:
            result["done"] = True

    threading.Thread(target=_target, daemon=True).start()

    # Tail the log file — child processes write to it via inherited file descriptors,
    # so this is more reliable than a QueueHandler across process boundaries.
    output = ""
    log_file = None
    while not result["done"]:
        time.sleep(1.0)
        if log_file is None and os.path.exists(log_path):
            log_file = open(log_path, "r")
        if log_file is not None:
            output += log_file.read()
        yield output

    # Final read after the thread finishes
    if log_file is None and os.path.exists(log_path):
        log_file = open(log_path, "r")
    if log_file is not None:
        output += log_file.read()
        log_file.close()

    if result["error"]:
        output += f"\n--- FAILED ---\n{result['error']}"
    else:
        output += "\n--- Complete ---"

    yield output


def run_detect(weight, source, project, conf_thres, iou_thres, device, img_size, dy_thres):
    if not weight.strip():
        yield "ERROR: Model weights path is required."
        return
    if not source.strip():
        yield "ERROR: Source images directory is required."
        return
    if not project.strip():
        yield "ERROR: Project directory is required."
        return

    opt = argparse.Namespace(
        backbone_detector="yolov7-w6",
        weight=weight.strip(),
        source=source.strip(),
        num_classes=1,
        img_size=int(img_size),
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        device=device.strip(),
        view_img=False,
        save_txt=True,
        save_conf=True,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        project=project.strip(),
        name="exp",
        exist_ok=True,
        dy_thres=dy_thres,
    )

    import partinet.DynamicDet.detect as _m

    # Remove stale FileHandlers left by previous detect() calls
    detect_logger = logging.getLogger("partinet_detect")
    for h in detect_logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            detect_logger.removeHandler(h)
            h.close()

    yield from _stream_stage(
        _m.detect,
        (opt,),
        loggers=["partinet_detect"],
    )


def run_star(labels, images, output, conf, relion, relion_project_dir, mrc_prefix):
    relion_project_dir = (relion_project_dir or "").strip()
    mrc_prefix = (mrc_prefix or "").strip()

    if not labels.strip():
        yield "ERROR: Labels directory is required."
        return
    if not images.strip():
        yield "ERROR: Images directory is required."
        return
    if not output.strip():
        yield "ERROR: Output STAR file path is required."
        return
    if relion and not relion_project_dir:
        yield "ERROR: RELION project directory is required when RELION output is enabled."
        return

    import partinet.process_utils.star_file as _m
    yield from _stream_stage(
        _m.main,
        (
            labels.strip(), images.strip(), output.strip(), conf,
            relion,
            relion_project_dir or None,
            mrc_prefix,
        ),
        loggers=["partinet.process_utils.star_file"],
    )


# ── Star File analysis ───────────────────────────────────────────────────────

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


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
    from PIL import Image as _PIL
    try:
        with _PIL.open(path) as img:
            return img.size  # (w, h)
    except Exception:
        return 4096, 4096


def load_detections(labels_dir, images_dir):
    labels_dir, images_dir = labels_dir.strip(), images_dir.strip()
    _empty = (None, "", None, None, None, gr.update(choices=[]), "")

    if not labels_dir or not images_dir:
        return (None, "Both directories are required.", None, None, None, gr.update(choices=[]), "")
    if not os.path.isdir(labels_dir):
        return (None, f"Labels directory not found: `{labels_dir}`", None, None, None, gr.update(choices=[]), "")
    if not os.path.isdir(images_dir):
        return (None, f"Images directory not found: `{images_dir}`", None, None, None, gr.update(choices=[]), "")

    label_files = sorted(f for f in os.listdir(labels_dir) if f.endswith(".txt"))
    if not label_files:
        return (None, "No .txt label files found in labels directory.", None, None, None, gr.update(choices=[]), "")

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
        return (None, "No label files had matching images in the images directory.", None, None, None, gr.update(choices=[]), "")

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
        _count_plot(mics, DEFAULT),
        gr.update(choices=choices, value=choices[0]),
        _retained_text(all_confs, DEFAULT),
    )


def _conf_plot(confs, threshold):
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
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(sizes_px, bins=50, color="darkorange", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Box size (px)")
    ax.set_ylabel("Detections")
    ax.set_title("Box size distribution")
    fig.tight_layout()
    return fig


def _count_plot(mics, threshold):
    counts = [sum(1 for d in m["dets"] if d["conf"] >= threshold) for m in mics]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.hist(counts, bins=max(1, min(40, len(set(counts)))), color="mediumseagreen", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Particles per micrograph")
    ax.set_ylabel("Micrographs")
    ax.set_title(f"Particle count distribution  (threshold = {threshold:.2f})")
    fig.tight_layout()
    return fig


def _retained_text(confs, threshold):
    total = len(confs)
    kept = sum(1 for c in confs if c >= threshold)
    pct = 100 * kept / total if total > 0 else 0
    return f"**{kept:,} / {total:,} particles retained** ({pct:.1f}%)"


def _draw_detections(mic, threshold, max_px=1000):
    from PIL import Image as _PIL, ImageDraw
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
    count_fig = _count_plot(state["mics"], threshold)
    img, mic_stats = None, ""
    if mic_name:
        mic = next((m for m in state["mics"] if m["name"] == mic_name), None)
        if mic:
            img, n = _draw_detections(mic, threshold)
            mic_stats = f"**{n}** particles shown"
    return retained, conf_fig, count_fig, img, mic_stats


def update_micrograph(state, mic_name, threshold):
    if state is None or not mic_name:
        return None, ""
    mic = next((m for m in state["mics"] if m["name"] == mic_name), None)
    if mic is None:
        return None, ""
    img, n = _draw_detections(mic, threshold)
    return img, f"**{n}** particles shown"


# ── Helpers for global project directory ─────────────────────────────────────

def _latest_labels_dir(project_dir):
    import glob as _glob
    exp_dirs = sorted(
        [d for d in _glob.glob(os.path.join(project_dir, "exp*")) if os.path.isdir(d)],
        key=os.path.getmtime,
        reverse=True,
    )
    for d in exp_dirs:
        candidate = os.path.join(d, "labels")
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(project_dir, "exp", "labels")


def update_project_dir(project_dir):
    p = (project_dir or "").strip()
    if not p:
        return ("",) * 6
    denoised = os.path.join(p, "denoised")
    labels = _latest_labels_dir(p) if os.path.isdir(p) else os.path.join(p, "exp", "labels")
    return (
        p,                                  # d1_project
        denoised,                           # d2_source
        p,                                  # d2_project
        labels,                             # d3_labels
        denoised,                           # d3_images
        os.path.join(p, "particles.star"),  # d3_output
    )


# ── Gradio UI ────────────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="PartiNet") as app:
        gr.Markdown(
            "# PartiNet\n"
            "Automated cryo-EM particle picker. "
            "Run each stage in order: **1 · Denoise → 2 · Detect → 3 · Star File**."
        )

        _testing_project = "/vast/cryoem/cryoem_scratch/lab_shakeel/perera.m/EMPIAR_10089/gui_testing"
        gr_project = gr.Textbox(
            label="Project directory",
            value=_testing_project,
            placeholder="/path/to/my_project",
            info="Set once — auto-fills project paths in all three stages below",
        )

        with gr.Tabs():

            # ── 1. Denoise ───────────────────────────────────────────────────
            with gr.Tab("1 · Denoise"):
                gr.Markdown(
                    "Improve signal-to-noise in raw micrographs using a Wiener filter. "
                    "Output images are saved to `project/denoised/`."
                )
                with gr.Row():
                    d1_source = gr.Textbox(
                        label="Raw micrographs directory",
                        placeholder="/path/to/motion_corrected",
                        value="/vast/cryoem/cryoem_scratch/lab_shakeel/perera.m/EMPIAR_10089/motioncorrected",
                        info="Folder of .mrc files from RELION or CryoSPARC motion correction",
                    )
                    d1_project = gr.Textbox(
                        label="Project directory",
                        placeholder="/path/to/my_project",
                        value="/vast/cryoem/cryoem_scratch/lab_shakeel/perera.m/EMPIAR_10089/gui_testing",
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
                        value=8,
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
                d1_btn.click(
                    run_denoise,
                    inputs=[d1_source, d1_project, d1_fmt, d1_workers],
                    outputs=d1_log,
                )

            # ── 2. Detect ────────────────────────────────────────────────────
            with gr.Tab("2 · Detect"):
                gr.Markdown(
                    "Locate particles in denoised micrographs using the DynamicDet model. "
                    "Results are saved to `project/exp/`."
                )
                with gr.Row():
                    d2_weight = gr.Textbox(
                        label="Model weights (.pt)",
                        placeholder="/path/to/model.pt",
                        value="/stornext/System/data/software/rhel/9/base/structbio/PartiNet/weights/denoised_micrographs_v2.pt",
                        info="Pre-trained weights — download from HuggingFace or use your own trained model",
                    )
                    d2_source = gr.Textbox(
                        label="Denoised images directory",
                        placeholder="/path/to/my_project/denoised",
                        value="/vast/cryoem/cryoem_scratch/lab_shakeel/perera.m/EMPIAR_10089/gui_testing/denoised",
                        info="Output of the Denoise step (project/denoised/)",
                    )
                with gr.Row():
                    d2_project = gr.Textbox(
                        label="Project directory",
                        placeholder="/path/to/my_project",
                        value="/vast/cryoem/cryoem_scratch/lab_shakeel/perera.m/EMPIAR_10089/gui_testing",
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
                d2_btn.click(
                    run_detect,
                    inputs=[d2_weight, d2_source, d2_project, d2_conf, d2_iou, d2_device, d2_imgsize, d2_dy],
                    outputs=d2_log,
                )

            # ── 3. Star File ─────────────────────────────────────────────────
            with gr.Tab("3 · Star File"):
                gr.Markdown(
                    "Load detections from a Detect run, explore statistics, set a confidence "
                    "threshold interactively, then generate a STAR file for **CryoSPARC** or **RELION**."
                )

                # --- Load ---
                with gr.Row():
                    d3_labels = gr.Textbox(
                        label="Labels directory",
                        placeholder="/path/to/project/exp/labels",
                        value="/vast/cryoem/cryoem_scratch/lab_shakeel/perera.m/EMPIAR_10089/gui_testing/exp/labels",
                        info="Folder of .txt detection files from the Detect step",
                    )
                    d3_images = gr.Textbox(
                        label="Denoised images directory",
                        placeholder="/path/to/project/denoised",
                        value="/vast/cryoem/cryoem_scratch/lab_shakeel/perera.m/EMPIAR_10089/gui_testing/denoised",
                        info="Used to resolve image dimensions for coordinate conversion",
                    )
                d3_state = gr.State(None)
                d3_load_btn = gr.Button("Load Detections", variant="secondary")
                d3_summary = gr.Markdown("")

                # --- Statistics ---
                with gr.Row():
                    d3_conf_plot = gr.Plot(format="png",label="Confidence distribution")
                    d3_size_plot = gr.Plot(format="png",label="Box size distribution")
                d3_count_plot = gr.Plot(format="png",label="Particles per micrograph")

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
                    value="/vast/cryoem/cryoem_scratch/lab_shakeel/perera.m/EMPIAR_10089/gui_testing/particles.star",
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
                    outputs=[d3_state, d3_summary, d3_conf_plot, d3_size_plot, d3_count_plot, d3_mic_select, d3_retained],
                )
                d3_thresh.change(
                    update_threshold,
                    inputs=[d3_state, d3_thresh, d3_mic_select],
                    outputs=[d3_retained, d3_conf_plot, d3_count_plot, d3_preview, d3_mic_stats],
                )
                d3_mic_select.change(
                    update_micrograph,
                    inputs=[d3_state, d3_mic_select, d3_thresh],
                    outputs=[d3_preview, d3_mic_stats],
                )
                d3_star_btn.click(
                    run_star,
                    inputs=[d3_labels, d3_images, d3_output, d3_thresh, d3_relion, d3_relion_dir, d3_mrc_prefix],
                    outputs=d3_log,
                )

        gr_project.change(
            update_project_dir,
            inputs=[gr_project],
            outputs=[d1_project, d2_source, d2_project, d3_labels, d3_images, d3_output],
        )

    return app


def launch_gui(host="0.0.0.0", port=7860, share=False):
    app = build_app()
    app.queue()
    app.launch(server_name=host, server_port=port, share=share, theme=gr.themes.Soft(), ssr_mode=False)
