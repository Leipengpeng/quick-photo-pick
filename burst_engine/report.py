"""Write metadata and browser presentation for the photo selector."""

import json
from pathlib import Path


QUALITY_GALLERY = "quality-samples.html"
ANNOTATIONS = {
    ("blink", "closed"): ("blink", "闭眼"),
    ("blink", "open"): ("blink", "睁眼"),
    ("motion_blur", "blurred"): ("motion", "模糊"),
    ("motion_blur", "sharp"): ("motion", "清晰"),
    ("motion", "blurred"): ("motion", "模糊"),
    ("motion", "sharp"): ("motion", "清晰"),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def frame_preview(frame, guide, target, name, engine_read, copies):
    from PIL import Image
    from export_photos import local_link

    information = guide.get(Path(frame["path"]).name, {})
    display = dict(frame)
    display["filename"] = Path(frame["path"]).name
    for key in ("source", "source_label", "chapter", "role", "sequence", "scene", "capture_at"):
        if key in information:
            display[key] = information[key]
    display["src"] = None
    display["annotation"] = None
    if not frame.get("analyzed_path"):
        return display

    original = copies[frame["analyzed_path"]]
    if original.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        with Image.open(original) as image:
            width, height = image.size
            if image.getexif().get(274, 1) in (5, 6, 7, 8):
                width, height = height, width
        preview = original
    else:
        image = Image.fromarray(engine_read(original, 2560))
        preview = target / "previews" / f"{name}.jpg"
        preview.parent.mkdir(exist_ok=True)
        image.save(preview, quality=95, subsampling=0)
        width, height = image.size
    display["src"] = local_link(preview, target)
    display["width"], display["height"] = width, height

    reference = information.get("reference")
    if isinstance(reference, dict):
        label = ANNOTATIONS.get((reference.get("kind"), reference.get("expected")))
        if label:
            annotation_type, caption = label
            display["annotation"] = {
                "type": annotation_type,
                "label": caption,
                "reference": information.get("source"),
            }
    return display


def build_payload(result, target, summary, guide, exported):
    import cv2

    from engine import read_image
    from export_photos import capture_order, local_link

    cv2.setNumThreads(1)
    scene_by_group = {
        group_id: item["scene_id"] for item in result["scene_clusters"] for group_id in item["groups"]
    }
    photo_notes = guide.get("photos", {})
    groups = []
    for group in result["groups"]:
        group_id = group["group"]
        ranked = sorted(group["frames"], key=capture_order)
        frames = [
            frame_preview(frame, photo_notes, target, f"{group_id:03}-{index:03}", read_image, exported["files"])
            for index, frame in enumerate(ranked, 1)
        ]
        exemplar = photo_notes.get(frames[0]["filename"], {})
        groups.append({
            "id": group_id, "group": group_id, "scene": scene_by_group[group_id],
            "capture_time": ranked[0].get("timestamp"), "title": exemplar.get("title", f"第{group_id:02d}组"),
            "source": exemplar.get("source_label"), "source_label": exemplar.get("source_label"),
            "chapter": exemplar.get("chapter", "拍摄时间线"), "role": exemplar.get("role", "capture"),
            "kind": (exemplar.get("reference") or {}).get("kind"),
            "basis": group["basis"], "uncertain_boundary": group["uncertain_boundary"],
            "frames": frames, "keeper_target": group["keeper_target"],
            **exported["groups"][group_id],
        })
    scenes = []
    for item in result["scene_clusters"]:
        member_ids = item["groups"]
        links = [link for link in result["scene_links"] if set(link["groups"]).issubset(member_ids)]
        scenes.append({"id": item["scene_id"], "groups": member_ids, "links": links})
    return {
        "title": guide.get("title", "摄影连拍审片"),
        "description": guide.get("description"), "generated_at": summary["generated_at"],
        "run_id": summary["run_id"], "rehearsal": summary["rehearsal"], "summary": summary,
        "groups": groups, "scenes": scenes, "evidence": result["scene_links"], "source_urls": guide.get("source_urls", {}),
        "grouped_folder": local_link(exported["folder"], target, directory=True),
        "quality_gallery": QUALITY_GALLERY if (target / QUALITY_GALLERY).is_file() else None,
    }
