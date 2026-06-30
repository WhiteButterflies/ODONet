"""
Experiment 2 (test-time): search -> template attention probe + visualization.

Why a hook on `attn_drop`:
    In FastITPN's `Attention.forward`, the post-softmax attention is a local
    variable consumed by `attn @ v` and never returned, so a forward hook on the
    `Attention` module only sees the projected output. But the attention is
    passed through the `attn.attn_drop` (nn.Dropout) submodule right after the
    softmax. At eval() dropout is identity, so a forward hook on `attn_drop`
    captures the *exact* attention matrix used in `attn @ v` -- no re-computation,
    and relative-position bias is already included.

Token layout inside the joint (main) blocks at test time:
    [ search(num_patch_x) ; template(num_template * num_patch_z) ; sam(1) ]
    search comes first, templates are stacked, the last token is the SAM embed.
    => search->template attention = attn[:, qs, kt] with
       qs = [0 : num_search], kt = [num_search : num_search + num_tpl_tokens].

Only the main blocks (`encoder.body.blocks.{8..31}`) carry `attn.attn_drop`;
the stage1/stage2 blocks have `attn = None`, so name-filtering is unambiguous.
"""
import os
import re

import cv2
import numpy as np
import torch


class AttnProbe:
    """Forward-hook the post-softmax attention of every main joint-attention
    block (`*.blocks.{i}.attn.attn_drop`) and stash it per layer for batch 0."""

    _NAME_RE = re.compile(r"\.blocks\.(\d+)\.attn\.attn_drop$")

    def __init__(self, network, block_indices=None, verbose=True):
        self.store = {}          # name -> (H, N, N) for batch index 0
        self.handles = []
        self._idx = {}           # name -> block index
        # negative attention redirection (erase) state — off by default.
        # Levels (cumulative): 1 ROI rows don't read template; 2 + ROI rows don't read
        # ROI self; 3 + nobody reads ROI keys; 4 + neck Injector/Extractor masks;
        # 5 + feature erase on xs (done by the tracker). See set_erase_from_rois.
        self.erase_active = False
        self.erase_rows = None      # ROI search-cell indices (query rows / key cols)
        self.erase_tpl_cols = None  # template key-column indices (fg-only or all)
        self.erase_level = 0
        self.erase_layers = set()   # joint-block indices to apply the mask at
        self.neck_handles = []      # forward-pre-hooks on injector/extractor MHA
        matched = []
        for name, module in network.named_modules():
            m = self._NAME_RE.search(name)
            if m is None:
                continue
            bi = int(m.group(1))
            if block_indices is not None and bi not in block_indices:
                continue
            self._idx[name] = bi
            self.handles.append(module.register_forward_hook(self._make_hook(name)))
            matched.append((bi, name))
        if verbose:
            names = [n for _, n in sorted(matched)]
            print("[AttnProbe] hooked %d attn_drop modules: %s" % (len(names), names))
        # neck Injector/Extractor MHA pre-hooks (active only at erase level >= 4)
        n_neck = 0
        for name, module in network.named_modules():
            if re.search(r"\.injector\.attn$", name):
                self.neck_handles.append(module.register_forward_pre_hook(self._injector_pre_hook))
                n_neck += 1
            elif re.search(r"\.(extractor|extra_extractors\.\d+)\.attn$", name):
                self.neck_handles.append(module.register_forward_pre_hook(self._extractor_pre_hook))
                n_neck += 1
        if verbose:
            print("[AttnProbe] hooked %d neck injector/extractor MHA modules" % n_neck)

    def _make_hook(self, name):
        bi = self._idx[name]
        def hook(module, inp, out):
            # out: (B, H, N, N); eval-mode dropout => out == softmax attention.
            if self.erase_active and self.erase_rows and bi in self.erase_layers:
                out = out.clone()
                roi = torch.as_tensor(self.erase_rows, device=out.device)
                if self.erase_level >= 3:                     # nobody reads ROI keys
                    out[:, :, :, roi] = 0.0
                if self.erase_tpl_cols:                        # L1: ROI rows don't read template
                    tcol = torch.as_tensor(self.erase_tpl_cols, device=out.device)
                    out[:, :, roi.view(-1, 1), tcol.view(1, -1)] = 0.0
                if self.erase_level >= 2:                      # L2: ROI rows don't read ROI self
                    out[:, :, roi.view(-1, 1), roi.view(1, -1)] = 0.0
                out = out / out.sum(-1, keepdim=True).clamp_min(1e-8)  # renorm affected rows
                self.store[name] = out.detach()[0]
                return out  # replace the attention used by `attn @ v`
            self.store[name] = out.detach()[0]  # (H, N, N), batch 0
            return None
        return hook

    def _injector_pre_hook(self, module, args):
        # injector.attn(query=x_full, key=xs_search, value=xs_search):
        # level>=4 -> no query may read the ROI search key cols.
        if not (self.erase_active and self.erase_level >= 4 and self.erase_rows):
            return None
        q, k, v = args[0], args[1], args[2]
        Lk = k.shape[0]
        roi = [c for c in self.erase_rows if c < Lk]
        if not roi:
            return None
        mask = q.new_zeros(q.shape[0], Lk)
        mask[:, roi] = float('-inf')
        return (q, k, v, None, True, mask)  # ..., key_padding_mask, need_weights, attn_mask

    def _extractor_pre_hook(self, module, args):
        # extractor.attn(query=xs_search, key=x_full, value=x_full): level>=4 ->
        # ROI rows don't read template cols; no row reads the ROI search cols of x.
        if not (self.erase_active and self.erase_level >= 4 and self.erase_rows):
            return None
        q, k, v = args[0], args[1], args[2]
        Lq, Lk = q.shape[0], k.shape[0]
        mask = q.new_zeros(Lq, Lk)
        roi = torch.as_tensor(self.erase_rows, device=q.device)
        roi_kcols = roi[roi < Lk]
        if roi_kcols.numel():
            mask[:, roi_kcols] = float('-inf')               # nobody reads ROI's portion of x
        roi_rows = roi[roi < Lq]
        tcol = torch.as_tensor([c for c in (self.erase_tpl_cols or []) if c < Lk], device=q.device)
        if roi_rows.numel() and tcol.numel():
            mask[roi_rows.view(-1, 1), tcol.view(1, -1)] = float('-inf')  # ROI doesn't read template
        return (q, k, v, None, True, mask)

    def set_erase_from_rois(self, roi_boxes, fx, num_search, fg_weight, level, layers,
                            tpl_mode="aggressive", fg_thresh=0.5):
        """Enable layered erase from drawn ROI boxes. tpl_mode 'aggressive' masks all
        template key cols, 'conservative' masks only fg (target) cols. Returns
        (roi_rows, tpl_cols). Level 5's feature erase is applied by the tracker."""
        rows = sorted({c for b in (roi_boxes or []) for c in _cells_in_box(b, fx)})
        if str(tpl_mode) == "conservative":
            fg_idx = np.where(np.asarray(fg_weight) > float(fg_thresh))[0]
            tpl_cols = (int(num_search) + fg_idx).tolist()
        else:
            tpl_cols = list(range(int(num_search), int(num_search) + len(fg_weight)))
        self.erase_rows = rows
        self.erase_tpl_cols = tpl_cols
        self.erase_level = int(level)
        self.erase_layers = set(int(l) for l in (layers or []))
        self.erase_active = bool(rows) and self.erase_level >= 1 and bool(self.erase_layers)
        return rows, tpl_cols

    def clear_erase(self):
        self.erase_active = False
        self.erase_rows = None
        self.erase_tpl_cols = None
        self.erase_level = 0
        self.erase_layers = set()

    def clear(self):
        self.store.clear()

    def remove(self):
        for h in self.handles + self.neck_handles:
            h.remove()
        self.handles = []
        self.neck_handles = []

    def layer_names(self):
        """Layer names sorted by block index."""
        return [n for n, _ in sorted(self._idx.items(), key=lambda kv: kv[1])]


def compute_template_fg_weight(network, template_list, template_anno_list):
    """Per-template-patch foreground weight, replicating the model's own
    `z_indicate_mask` (see fastitpn.prepare_tokens_with_masks). Returns a
    (num_template * num_patch_z,) float array in [0, 1], near 1 == foreground."""
    body = network.encoder.body
    ps = body.patch_size
    weights = []
    with torch.no_grad():
        for img, anno in zip(template_list, template_anno_list):
            anno = anno.reshape(-1)[:4].unsqueeze(0).to(img.device)       # (1,4) xywh-norm
            mask = body.create_mask(img, anno)                            # (1, H, W)
            mask = mask.unfold(1, ps, ps).unfold(2, ps, ps).mean(dim=(3, 4)).flatten(1)
            weights.append(mask[0])                                       # (num_patch_z,)
    return torch.cat(weights, dim=0).float().cpu().numpy()


# ----------------------------------------------------------------------------- #
# rendering helpers
# ----------------------------------------------------------------------------- #
def _overlay(base_bgr, heat, alpha=0.5, box=None, vmin=None, vmax=None):
    """Overlay a heatmap. With vmin/vmax given, normalize against that shared
    range (comparable across panels); otherwise per-heat min-max (relative)."""
    h, w = base_bgr.shape[:2]
    hm = heat.astype(np.float32)
    lo = hm.min() if vmin is None else vmin
    hi = hm.max() if vmax is None else vmax
    hm = np.clip((hm - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    hm = (hm * 255).astype(np.uint8)
    hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_CUBIC)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    out = cv2.addWeighted(base_bgr, 1 - alpha, hm, alpha, 0)
    if box is not None:
        x, y, bw, bh = [float(v) for v in box]
        cv2.rectangle(out, (int(x * w), int(y * h)),
                      (int((x + bw) * w), int((y + bh) * h)), (255, 255, 255), 1)
    return out


def _bar_panel(values, height, width=160, labels=None):
    """A small bar chart of each template's share of the ROI's total
    template-attention (bars sum to 100%, height scaled to the max bar)."""
    img = np.full((height, width, 3), 40, np.uint8)
    v = np.asarray(values, np.float32)
    share = v / (v.sum() + 1e-8)
    n = len(v)
    pad = 6
    bw = max((width - pad * (n + 1)) // n, 4)
    base_y, top = height - 16, height - 16 - 14
    hmax = share.max() + 1e-8
    cv2.putText(img, "tpl share", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
    for i in range(n):
        x0 = pad + i * (bw + pad)
        bh = int((share[i] / hmax) * top)
        cv2.rectangle(img, (x0, base_y - bh), (x0 + bw, base_y), (0, 165, 255), -1)
        cv2.putText(img, labels[i] if labels else "z%d" % i, (x0, height - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "%d%%" % int(round(share[i] * 100)), (x0, max(base_y - bh - 2, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _label(img, text, top=False, color=(255, 255, 255)):
    img = np.ascontiguousarray(img)
    y = 14 if top else img.shape[0] - 6
    cv2.putText(img, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def _zlabel(t, frame_ids=None):
    """Template label, with its source frame number when available: 'z0 f37'."""
    if frame_ids is not None and t < len(frame_ids):
        return "z%d f%d" % (t, int(frame_ids[t]))
    return "z%d" % t


def _cells_in_box(box, fx):
    """Flat search-grid indices whose cell center falls inside a normalized
    xywh box; falls back to the single nearest cell for a tiny box."""
    x, y, bw, bh = box
    cells = []
    for gy in range(fx):
        for gx in range(fx):
            cx, cy = (gx + 0.5) / fx, (gy + 0.5) / fx
            if x <= cx <= x + bw and y <= cy <= y + bh:
                cells.append(gy * fx + gx)
    if not cells:
        cx, cy = x + bw / 2.0, y + bh / 2.0
        gx = min(max(int(cx * fx), 0), fx - 1)
        gy = min(max(int(cy * fx), 0), fx - 1)
        cells = [gy * fx + gx]
    return cells


def select_search_rois(search_bgr, window="draw search ROI (drag; Enter=add, Esc=done)"):
    """Interactive ROI selection on the search crop, drawn *before* the forward.
    Returns a list of normalized (x, y, w, h) boxes; empty if nothing is drawn
    or if there is no GUI backend."""
    h, w = search_bgr.shape[:2]
    try:
        boxes = cv2.selectROIs(window, search_bgr, showCrosshair=True, fromCenter=False)
    except Exception as e:
        print("[AttnProbe] selectROIs unavailable (%s); skipping ROI this frame." % e)
        return []
    try:
        cv2.destroyWindow(window)
    except Exception:
        pass
    out = []
    for b in boxes:
        x, y, bw, bh = [float(v) for v in b]
        if bw > 0 and bh > 0:
            out.append((x / w, y / h, bw / w, bh / h))
    return out


def _draw_markers(img, fx, points=None, rois=None):
    h, w = img.shape[:2]
    out = img.copy()
    for i, (x, y, bw, bh) in enumerate(rois or []):
        cv2.rectangle(out, (int(x * w), int(y * h)),
                      (int((x + bw) * w), int((y + bh) * h)), (0, 255, 0), 2)
        cv2.putText(out, "roi%d" % i, (int(x * w) + 2, int(y * h) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    for label, (cx, cy) in (points or {}).items():
        cxx, cyy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        cv2.circle(out, (int(cxx * w), int(cyy * h)), 4, (0, 0, 255), -1)
    return out


def _render_query_panel(S2T, q_indices, label, num_tpl, fz, template_bgr_list,
                        template_box_list, fg_weight, save_dir, frame_id,
                        alpha=0.5, extra="", global_norm=False, frame_ids=None):
    """Aggregate attention over the given search cells and render its map to each
    template (per-template OR globally-shared color scale), append a per-template
    attention-share bar, and report the FG/BG split for the aggregated query."""
    s2t_q = S2T[q_indices].mean(0)                       # mean over the query cells
    grid = s2t_q.reshape(num_tpl, fz, fz)
    vlo = float(grid.min()) if global_norm else None     # shared scale across templates
    vhi = float(grid.max()) if global_norm else None
    cells = [_label(_overlay(template_bgr_list[t], grid[t], alpha,
                             box=template_box_list[t], vmin=vlo, vmax=vhi), _zlabel(t, frame_ids))
             for t in range(num_tpl)]
    totals = grid.reshape(num_tpl, -1).sum(1)            # raw attention mass per template
    bar = _bar_panel(totals, template_bgr_list[0].shape[0],
                     labels=["z%d" % t for t in range(num_tpl)])
    row = np.concatenate(cells + [bar], axis=1)
    fgw = float((s2t_q * fg_weight).sum())
    bgw = float((s2t_q * (1.0 - fg_weight)).sum())
    tot = fgw + bgw + 1e-8
    tag = "%s FG=%.2f BG=%.2f%s%s" % (label, fgw / tot, bgw / tot, extra,
                                      " [gnorm]" if global_norm else "")
    row = _label(row, tag, top=True)
    cv2.imwrite(os.path.join(save_dir, "%04d_q-%s.png" % (frame_id, label)), row)
    share = totals / (totals.sum() + 1e-8)
    print("[AttnProbe] frame %d  %s  tpl_share=%s" % (
        frame_id, tag, np.array2string(share, precision=2, separator=",")))


def render_search_to_template(store, layer_names, num_search, fx, num_tpl, fz,
                              template_bgr_list, template_box_list, search_bgr,
                              query_points, fg_weight, save_dir, frame_id,
                              roi_regions=None, global_norm=False, frame_ids=None, alpha=0.5):
    """
    store           : {layer_name: (H, N, N)} captured by AttnProbe
    layer_names     : which layers to average over (e.g. probe.layer_names())
    num_search, fx  : number of search tokens and the search grid width (196, 14)
    num_tpl, fz     : number of templates and the template grid width (5, 7)
    template_bgr_*  : list of BGR uint8 template crops and their xywh-norm boxes
    search_bgr      : BGR uint8 search crop
    query_points    : {label: (cx, cy)} auto query points (target / distractor)
    roi_regions     : list of normalized (x, y, w, h) hand-drawn ROIs; each is
                      aggregated over every search cell it covers
    fg_weight       : (num_tpl*fz*fz,) foreground weight in [0,1]
    """
    mats = [store[n] for n in layer_names if n in store]
    if not mats:
        print("[AttnProbe] store empty -- nothing to render (frame %d)" % frame_id)
        return
    os.makedirs(save_dir, exist_ok=True)

    ntpl_tok = num_tpl * fz * fz
    A = torch.stack(mats, 0).mean(0).float()                 # (H, N, N) avg over layers
    N = A.shape[-1]
    end = num_search + ntpl_tok
    if end > N:
        print("[AttnProbe] layout mismatch: need cols up to %d but N=%d" % (end, N))
        return
    S2T = A.mean(0)[:num_search, num_search:end].cpu().numpy()  # (num_search, ntpl_tok), head-mean

    # ---- search-space attention to FG-typed vs BG-typed template patches ----
    w = fg_weight.reshape(1, -1)
    fg_map = (S2T * w).sum(1).reshape(fx, fx)
    bg_map = (S2T * (1.0 - w)).sum(1).reshape(fx, fx)
    diff = fg_map - bg_map
    # global_norm: share one scale between the FG and BG maps so colors are comparable
    fb_lo = min(float(fg_map.min()), float(bg_map.min())) if global_norm else None
    fb_hi = max(float(fg_map.max()), float(bg_map.max())) if global_norm else None
    search_marked = _draw_markers(search_bgr, fx, points=query_points, rois=roi_regions)
    panel = np.concatenate([
        _label(search_marked, "search"),
        _label(_overlay(search_bgr, fg_map, alpha, vmin=fb_lo, vmax=fb_hi), "->tpl FG"),
        _label(_overlay(search_bgr, bg_map, alpha, vmin=fb_lo, vmax=fb_hi), "->tpl BG"),
        _label(_overlay(search_bgr, diff, alpha), "FG - BG"),
    ], axis=1)
    cv2.imwrite(os.path.join(save_dir, "%04d_S2T_fgbg.png" % frame_id), panel)

    # ---- auto point queries (target / distractor) ----
    for label, (cx, cy) in (query_points or {}).items():
        valid = (0.0 <= cx <= 1.0) and (0.0 <= cy <= 1.0)
        cxx, cyy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        gx, gy = min(int(cxx * fx), fx - 1), min(int(cyy * fx), fx - 1)
        _render_query_panel(S2T, [gy * fx + gx], label, num_tpl, fz,
                            template_bgr_list, template_box_list, fg_weight,
                            save_dir, frame_id, alpha,
                            extra=" q=(%d,%d)%s" % (gy, gx, "" if valid else " [clamped]"),
                            global_norm=global_norm, frame_ids=frame_ids)

    # ---- hand-drawn ROI queries (aggregated over every cell inside the ROI) ----
    for i, box in enumerate(roi_regions or []):
        cells = _cells_in_box(box, fx)
        _render_query_panel(S2T, cells, "roi%d" % i, num_tpl, fz,
                            template_bgr_list, template_box_list, fg_weight,
                            save_dir, frame_id, alpha, extra=" (%d cells)" % len(cells),
                            global_norm=global_norm, frame_ids=frame_ids)


# ----------------------------------------------------------------------------- #
# template -> search (mirror direction): each template, foreground-weighted, as
# the query; where it attends in the search region.
# ----------------------------------------------------------------------------- #
def _draw_boxes(img, regions=None, roi_boxes=None):
    h, w = img.shape[:2]
    out = img.copy()
    for (name, box, color) in (regions or []):
        if box is None:
            continue
        x, y, bw, bh = [float(v) for v in box]
        cv2.rectangle(out, (int(x * w), int(y * h)),
                      (int((x + bw) * w), int((y + bh) * h)), color, 1)
    for b in (roi_boxes or []):
        x, y, bw, bh = [float(v) for v in b]
        cv2.rectangle(out, (int(x * w), int(y * h)),
                      (int((x + bw) * w), int((y + bh) * h)), (0, 255, 0), 1)
    return out


def _label_lines(img, lines, color=(255, 255, 255)):
    out = np.ascontiguousarray(img)
    y0 = out.shape[0] - 6 - 12 * (len(lines) - 1)
    for i, text in enumerate(lines):
        y = y0 + i * 12
        cv2.putText(out, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
    return out


def _grouped_bar_panel(totals, n_tpl, height, width=300):
    """Grouped bars: x = templates z0..z_{n-1}, bars per group = region totals
    (tgt/dis/roi), all scaled to the global max bar."""
    img = np.full((height, width, 3), 40, np.uint8)
    names = list(totals.keys())
    colors = {"tgt": (0, 0, 255), "dis": (255, 0, 0), "roi": (0, 255, 0)}
    allv = [v for vals in totals.values() for v in vals]
    vmax = (max(allv) if allv else 0.0) + 1e-8
    base_y, top = height - 18, height - 18 - 16
    group_w = max((width - 10) // max(n_tpl, 1), 12)
    bw = max(group_w // (len(names) + 1), 3)
    for t in range(n_tpl):
        gx0 = 6 + t * group_w
        for j, name in enumerate(names):
            x0 = gx0 + j * bw
            bh = int((totals[name][t] / vmax) * top)
            cv2.rectangle(img, (x0, base_y - bh), (x0 + bw - 1, base_y),
                          colors.get(name, (200, 200, 200)), -1)
        cv2.putText(img, "z%d" % t, (gx0, height - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    lx = 6
    for name in names:
        cv2.putText(img, name, (lx, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    colors.get(name, (200, 200, 200)), 1, cv2.LINE_AA)
        lx += 40
    return img


def render_template_to_search(store, layer_names, num_search, fx, num_tpl, fz,
                              search_bgr, fg_weight, target_box=None,
                              distractor_box=None, roi_boxes=None, save_dir=None,
                              frame_id=0, global_norm=False, frame_ids=None, alpha=0.5):
    """Each template (foreground-weighted) as query -> where it attends in the
    search region. Panels are arranged horizontally and labeled by source
    template; each is annotated with the total attention falling into the target
    / distractor / ROI regions, also summarized as a grouped bar."""
    mats = [store[n] for n in layer_names if n in store]
    if not mats:
        return
    os.makedirs(save_dir, exist_ok=True)
    ntpl_tok = num_tpl * fz * fz
    A = torch.stack(mats, 0).mean(0).float()
    N = A.shape[-1]
    end = num_search + ntpl_tok
    if end > N:
        print("[AttnProbe] T2S layout mismatch: need %d but N=%d" % (end, N))
        return
    T2S = A.mean(0)[num_search:end, :num_search].cpu().numpy()   # (ntpl_tok, num_search)

    # foreground-weighted aggregate of each template's query rows -> (num_tpl, num_search)
    fw = fg_weight.reshape(num_tpl, fz * fz)
    maps = np.zeros((num_tpl, num_search), np.float32)
    for t in range(num_tpl):
        rows = T2S[t * fz * fz:(t + 1) * fz * fz]                # (fz*fz, num_search)
        wt = fw[t]
        s = wt.sum()
        maps[t] = (rows * wt[:, None]).sum(0) / s if s > 1e-6 else rows.mean(0)
    grids = maps.reshape(num_tpl, fx, fx)
    vlo = float(grids.min()) if global_norm else None
    vhi = float(grids.max()) if global_norm else None

    # region cells + per-template totals
    region_cells = {}
    if target_box is not None:
        region_cells["tgt"] = _cells_in_box(target_box, fx)
    if distractor_box is not None:
        region_cells["dis"] = _cells_in_box(distractor_box, fx)
    roi_cells = sorted({c for b in (roi_boxes or []) for c in _cells_in_box(b, fx)})
    if roi_cells:
        region_cells["roi"] = roi_cells
    totals = {name: [float(maps[t][cells].sum()) for t in range(num_tpl)]
              for name, cells in region_cells.items()}

    regions = []
    if target_box is not None:
        regions.append(("tgt", target_box, (0, 0, 255)))
    if distractor_box is not None:
        regions.append(("dis", distractor_box, (255, 0, 0)))

    panels = [_label(_draw_boxes(search_bgr, regions, roi_boxes), "search")]
    for t in range(num_tpl):
        ov = _draw_boxes(_overlay(search_bgr, grids[t], alpha, vmin=vlo, vmax=vhi),
                         regions, roi_boxes)
        lines = ["%s->search" % _zlabel(t, frame_ids)] + ["%s=%.3f" % (n, totals[n][t]) for n in totals]
        panels.append(_label_lines(ov, lines))
    if totals:
        panels.append(_grouped_bar_panel(totals, num_tpl, search_bgr.shape[0]))
    strip = np.concatenate(panels, axis=1)
    cv2.imwrite(os.path.join(save_dir, "%04d_T2S.png" % frame_id), strip)
    print("[AttnProbe] frame %d  T2S totals %s%s" % (
        frame_id, {k: ["%.3f" % x for x in v] for k, v in totals.items()},
        " [gnorm]" if global_norm else ""))


# ----------------------------------------------------------------------------- #
# boxhead response maps (pre/post Hanning) + negative-redirection before/after
# ----------------------------------------------------------------------------- #
def _to_2d(m, fx):
    if hasattr(m, "detach"):
        return m.detach().float().cpu().reshape(fx, fx).numpy()
    return np.asarray(m, dtype=np.float32).reshape(fx, fx)


def render_boxhead_responses(no_han_list, han_list, fx, search_bgr, save_dir, frame_id,
                             tag="", roi_boxes=None, alpha=0.5):
    """Save the two boxheads' pre/post-Hanning response maps as one strip overlaid
    on the search crop: [ bh0 pre | bh0 post | bh1 pre | bh1 post ].
    no_han_list / han_list each hold [boxhead0, boxhead1] score maps."""
    os.makedirs(save_dir, exist_ok=True)
    base = _draw_boxes(search_bgr, None, roi_boxes) if roi_boxes else search_bgr
    maps = [(no_han_list[0], "bh0 pre"), (han_list[0], "bh0 post"),
            (no_han_list[1], "bh1 pre"), (han_list[1], "bh1 post")]
    panels = [_label(_overlay(base, _to_2d(m, fx), alpha), lab) for m, lab in maps]
    strip = np.concatenate(panels, axis=1)
    cv2.imwrite(os.path.join(save_dir, "%04d_boxhead_resp%s.png" % (frame_id, tag)), strip)


def render_redirect_compare(clean_han_list, erased_han_list, fx, search_bgr, save_dir,
                            frame_id, roi_boxes=None, alpha=0.5):
    """Clean vs erased post-Hanning response per boxhead, plus the diff.
    Rows = boxhead0 / boxhead1; cols = clean | erased | (erased - clean)."""
    os.makedirs(save_dir, exist_ok=True)
    base = _draw_boxes(search_bgr, None, roi_boxes) if roi_boxes else search_bgr
    rows = []
    for bh in range(2):
        c = _to_2d(clean_han_list[bh], fx)
        e = _to_2d(erased_han_list[bh], fx)
        row = np.concatenate([
            _label(_overlay(base, c, alpha), "bh%d clean" % bh),
            _label(_overlay(base, e, alpha), "bh%d erased" % bh),
            _label(_overlay(base, e - c, alpha), "bh%d diff" % bh),
        ], axis=1)
        rows.append(row)
    cv2.imwrite(os.path.join(save_dir, "%04d_REDIRECT.png" % frame_id),
                np.concatenate(rows, axis=0))
