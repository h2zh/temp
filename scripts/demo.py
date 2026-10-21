"""Story demo against a running Crowd3D server: one visitor's photos give a facade, the crowd completes the building.

    python -m crowd3d.api --seed-demo          # in one terminal
    python scripts/demo.py                     # in another

Generates (or reuses) synthetic check-in photos around the demo landmark and
uploads them in stages: one visitor shoots the north side (-> facade model),
others cover more sides (-> more facade models), one more visitor completes
the ring (-> full model), and a few bad photos show the upload feedback.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any

import httpx

if __package__ in (None, ""):
    # `python scripts/demo.py` from a checkout: make the crowd3d package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crowd3d.synthetic import DEMO_LAT, DEMO_LON, DEMO_NAME, generate_dataset  # noqa: E402

DEFAULT_LANDMARK = "demo-building"
POLL_S = 2.0

REASONS = {
    "unreadable_image": "无法读取图片",
    "no_gps": "缺少GPS定位",
    "no_landmark": "附近没有地标",
    "too_far": "距离地标太远",
    "facing_away": "镜头没有朝向建筑",
    "too_small": "分辨率太低",
    "too_blurry": "照片太模糊",
    "duplicate": "重复照片",
    "too_large": "文件过大",
    "no_heading": "缺少指南针朝向",
    "magnetic_heading": "使用磁北朝向",
}
JOB_STATUS = {"queued": "排队中", "running": "重建中", "done": "完成", "failed": "失败"}
SIDE_NAMES = {"N": "北", "NE": "东北", "E": "东", "SE": "东南", "S": "南", "SW": "西南", "W": "西", "NW": "西北"}
# Where each of the 8 sector labels sits in a 3x3 text compass.
GRID = [["NW", "N", "NE"], ["W", None, "E"], ["SW", "S", "SE"]]


class DemoError(RuntimeError):
    pass


def parse_size(text: str) -> tuple[int, int]:
    w, _, h = text.lower().partition("x")
    return int(w), int(h)


# ---------------------------------------------------------------- server calls


def check(r: httpx.Response) -> Any:
    if r.status_code >= 400:
        raise DemoError(f"{r.request.method} {r.request.url.path} -> HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def ensure_landmark(client: httpx.Client, landmark_id: str) -> dict[str, Any]:
    r = client.get(f"/api/landmarks/{landmark_id}")
    if r.status_code == 404:
        name = DEMO_NAME if landmark_id == DEFAULT_LANDMARK else f"{DEMO_NAME} {landmark_id}"
        body = {"id": landmark_id, "name": name, "lat": DEMO_LAT, "lon": DEMO_LON, "radius_m": 150}
        created = client.post("/api/landmarks", json=body)
        if created.status_code != 409:   # 409: created concurrently, fine
            check(created)
        print(f"已创建地标 {landmark_id}")
        r = client.get(f"/api/landmarks/{landmark_id}")
    return check(r)


def upload(client: httpx.Client, photo_dir: Path, landmark_id: str, user: str,
           entries: list[dict[str, Any]], labels: list[str]) -> int:
    """Upload one visitor's photos in one request; print per-file feedback; return the number accepted."""
    files = [("files", (e["file"], (photo_dir / e["file"]).read_bytes(), "image/jpeg")) for e in entries]
    body = check(client.post("/api/photos", data={"user_id": user, "landmark_id": landmark_id}, files=files))
    ok = sum(r["accepted"] for r in body["results"])
    print(f"  {user} 上传 {len(entries)} 张，通过 {ok} 张")
    for r in body["results"]:
        if r["accepted"]:
            where = f"站在{side(labels[r['sector']])}侧，距中心 {r['distance_m']:.0f} m"
            notes = "；".join(REASONS.get(w, w) for w in r["warnings"])
            print(f"    ✓ {r['filename']:<22} {where}" + (f"（提示：{notes}）" if notes else ""))
        else:
            print(f"    ✗ {r['filename']:<22} " + "、".join(REASONS.get(c, c) for c in r["reasons"]))
    for job in body["triggered_jobs"]:
        print(f"    → 触发任务：{job_label(job, labels)}（{job['num_photos']} 张照片）")
    return ok


def wait_for_jobs(client: httpx.Client, landmark_id: str, labels: list[str], timeout_s: float,
                  finished_before: set[str]) -> dict[str, Any]:
    """Poll the landmark until no job is queued or running; print every status change of the other jobs."""
    seen: dict[str, str] = {}
    deadline = time.monotonic() + timeout_s
    while True:
        detail = check(client.get(f"/api/landmarks/{landmark_id}"))
        for job in reversed(detail["jobs"]):
            if job["id"] in finished_before or seen.get(job["id"]) == job["status"]:
                continue
            seen[job["id"]] = job["status"]
            line = f"  [{time.strftime('%H:%M:%S')}] {job_label(job, labels)}：{JOB_STATUS.get(job['status'], job['status'])}"
            if job["status"] == "done":
                line += f"，用时 {job['finished_at'] - job['started_at']:.1f} s"
                model = next((m for m in detail["models"] if (m["kind"], m["sector"]) == (job["kind"], job["sector"])),
                             None)
                if model:
                    line += f" — {model_line(model)}"
            elif job["status"] == "failed":
                line += f" — {job['error']}"
            print(line, flush=True)
        if not any(j["status"] in ("queued", "running") for j in detail["jobs"]):
            return detail
        if time.monotonic() > deadline:
            raise DemoError(f"jobs still running after {timeout_s:.0f} s")
        time.sleep(POLL_S)


def finished_jobs(detail: dict[str, Any]) -> set[str]:
    return {j["id"] for j in detail["jobs"] if j["status"] in ("done", "failed")}


# ---------------------------------------------------------------- text output


def side(label: str) -> str:
    return SIDE_NAMES.get(label, label)


def pad(text: str, width: int) -> str:
    """Left-align to a terminal width, counting CJK characters as two columns."""
    used = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - used)


def job_label(job: dict[str, Any], labels: list[str]) -> str:
    return "全景模型" if job["kind"] == "full" else f"{side(labels[job['sector']])}立面模型"


def model_line(m: dict[str, Any]) -> str:
    geo = "已地理配准" if m["georegistered"] else "未地理配准"
    extra = ""
    if m["alignment_rmse_m"] is not None:
        extra += f"，GPS 残差 {m['alignment_rmse_m']:.1f} m"
    if m["heading_mae_deg"] is not None:
        extra += f"，朝向误差 {m['heading_mae_deg']:.1f}°"
    return f"{m['num_registered']}/{m['num_photos']} 张注册，{m['num_points']} 个点，{geo}{extra}"


def compass_text(detail: dict[str, Any]) -> str:
    """3x3 text compass for 8 sectors (a plain list otherwise): accepted/needed photos, * = facade model."""
    need = detail["thresholds"]["face_min_photos"]
    cells = {s["label"]: pad(side(s["label"]), 4) + f" {s['count']}/{need}{'*' if s['has_face_model'] else ' '}"
             for s in detail["sectors"]}
    if len(detail["sectors"]) != 8:
        return "    " + "  ".join(cells.values())
    centre = f"[{detail['progress'] * 100:3.0f}%]"
    return "\n".join("    " + "   ".join(cells[k] if k else f"{centre:^9}" for k in row) for row in GRID)


def status_text(detail: dict[str, Any]) -> str:
    t = detail["thresholds"]
    bar = "#" * round(detail["progress"] * 24)
    ready = sum(s["state"] == "face_ready" for s in detail["sectors"])
    full = detail["latest_full_model"]
    if full:
        full_text = "全景模型已生成"
    elif detail["full_ready"]:
        full_text = "全景模型条件已满足"
    else:
        full_text = f"全景模型需要 {t['full_min_sectors']} 面就绪且 {t['full_min_photos']} 张照片（现在 {ready} 面、{detail['num_photos']} 张）"
    return (f"{compass_text(detail)}\n"
            f"    进度 [{bar:<24}] {detail['progress'] * 100:.0f}% · 照片 {detail['num_photos']} 张 · "
            f"共同完成者 {detail['num_contributors']} 人 · {full_text}\n"
            f"    （x/{t['face_min_photos']} = 该面已通过的照片 / 生成立面模型所需张数，* = 已有立面模型）")


# ---------------------------------------------------------------- dataset and story


def load_dataset(photo_dir: Path | None, size: tuple[int, int], per_sector: int, num_sectors: int,
                 lat: float, lon: float, seed: int) -> tuple[Path, dict[str, Any]]:
    """Generate the synthetic photos, or reuse them when the directory already holds a manifest."""
    if photo_dir is None:
        key = f"{size[0]}x{size[1]}-s{num_sectors}-p{per_sector}-seed{seed}-{lat:.5f}-{lon:.5f}"
        photo_dir = Path(tempfile.gettempdir()) / "crowd3d-demo-photos" / key
    manifest_path = photo_dir / "manifest.json"
    if manifest_path.is_file():
        print(f"复用已生成的合成照片：{photo_dir}")
        return photo_dir, json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"生成合成照片（{size[0]}x{size[1]}，每面 {per_sector} 张）到 {photo_dir} ……", flush=True)
    t = time.monotonic()
    manifest = generate_dataset(photo_dir, landmark_lat=lat, landmark_lon=lon, num_sectors=num_sectors,
                                photos_per_sector=per_sector, size=size, seed=seed)
    print(f"  完成，{len(manifest['photos'])} 张，用时 {time.monotonic() - t:.0f} s")
    return photo_dir, manifest


def plan_stages(manifest: dict[str, Any], users: list[str], full_min_sectors: int,
                labels: list[str]) -> list[tuple[str, list[tuple[str, list[dict[str, Any]]]]]]:
    """[(title, [(user, entries)])]: north side by one visitor; more sides by others; the rest by one visitor.

    The last good-photo stage completes every remaining side in one request, so the
    full model is planned once, from all photos, instead of being rebuilt later.
    """
    good = [e for e in manifest["photos"] if e["kind"] == "good"]
    bad = [e for e in manifest["photos"] if e["kind"] != "good"]
    sectors = sorted({e["sector"] for e in good})
    by_sector = {s: [e for e in good if e["sector"] == s] for s in sectors}
    first, middle, last = sectors[:1], sectors[1:max(1, full_min_sectors - 1)], sectors[max(1, full_min_sectors - 1):]
    others = users[1:] or users

    def names(ss: list[int]) -> str:
        return "、".join(side(labels[s]) for s in ss)

    stages = [(f"{users[0]} 来打卡，在{names(first)}侧拍了 {len(by_sector[first[0]])} 张",
               [(users[0], by_sector[first[0]])])]
    if middle:
        batches: dict[str, list[dict[str, Any]]] = {}
        for i, s in enumerate(middle):
            batches.setdefault(others[i % len(others)], []).extend(by_sector[s])
        stages.append((f"更多游客加入，拍下了{names(middle)}几面", list(batches.items())))
    if last:
        finisher = users[(1 + len(middle)) % len(users)]
        stages.append((f"{finisher} 绕着建筑走了一圈，补齐了{names(last)}几面",
                       [(finisher, [e for s in last for e in by_sector[s]])]))
    if bad:
        stages.append(("并不是每张照片都能用：几张不合格的照片", [(others[0], bad)]))
    return stages


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python scripts/demo.py", description=__doc__.splitlines()[0])
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--landmark", default=DEFAULT_LANDMARK, help="landmark id (created at the demo location if missing)")
    ap.add_argument("--users", type=int, default=5, help="number of visitors taking part")
    ap.add_argument("--size", type=parse_size, default=(640, 480), help="photo size WxH (larger = slower SfM)")
    ap.add_argument("--per-sector", type=int, default=4, help="good photos per side")
    ap.add_argument("--seed", type=int, default=0, help="change it to upload a different set of photos")
    ap.add_argument("--photos", type=Path, default=None, help="dataset directory (reused if it has manifest.json)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for the jobs of one stage")
    args = ap.parse_args(argv)
    if args.users < 1:
        ap.error("--users must be at least 1")
    users = [f"游客{i + 1}" for i in range(args.users)]
    server = args.server.rstrip("/")

    with httpx.Client(base_url=server, timeout=httpx.Timeout(600.0, connect=5.0)) as client:
        try:
            health = check(client.get("/api/health"))
        except httpx.TransportError as e:
            print(f"无法连接 {server}（{e}）。请先启动服务：python -m crowd3d.api --seed-demo", file=sys.stderr)
            return 2
        try:
            detail = ensure_landmark(client, args.landmark)
            lm = detail
            t = detail["thresholds"]
            labels = [s["label"] for s in detail["sectors"]]
            print(f"服务器 {server}（重建器：{health['reconstructor']}）")
            print(f"地标：{lm['name']}（{lm['id']}）位于 {lm['lat']:.4f}, {lm['lon']:.4f}，半径 {lm['radius_m']:.0f} m")
            if detail["num_photos"]:
                print(f"注意：该地标已有 {detail['num_photos']} 张照片；同一组照片会被判为重复照片。"
                      f"可加 --seed 1 换一组照片，或用 --landmark 新建一个地标。")
            photo_dir, manifest = load_dataset(args.photos, args.size, args.per_sector, t["num_sectors"],
                                               lm["lat"], lm["lon"], args.seed)
            for i, (title, batches) in enumerate(plan_stages(manifest, users, t["full_min_sectors"], labels), 1):
                print(f"\n第 {i} 步：{title}")
                before = finished_jobs(detail)
                accepted = sum(upload(client, photo_dir, lm["id"], user, entries, labels) for user, entries in batches)
                detail = wait_for_jobs(client, lm["id"], labels, args.timeout, before)
                if accepted:
                    print(status_text(detail))
        except DemoError as e:
            print(f"出错：{e}", file=sys.stderr)
            return 1

    print("\n共同完成者：" + "，".join(f"{c['user_id']} {c['photos']} 张" for c in detail["contributors"]))
    full = detail["latest_full_model"]
    if full:
        print(f"全景模型：{model_line(full)}\n  点云：{server}{full['url']}")
    else:
        print("还没有全景模型（照片或已就绪的面不够）。")
    print(f"\n在浏览器中打开：{server}/landmark.html?id={lm['id']}")
    failed = [j for j in detail["jobs"] if j["status"] == "failed"]
    return 0 if full and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
