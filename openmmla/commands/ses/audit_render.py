"""The images and clips of a sensing audit (mmla ses-code --audit-render ID), made on the machine that
holds the recordings, in the environment that ran the VFA bases (cv2).

A frame is decoded as its base decoded it (vfa_base._process_keyframes): from the replay's sync time,
the keyframe interval added k times to the file's own offset, read through VideoStream in file mode
(int(time * fps), then CAP_PROP_POS_FRAMES), rotated and remapped as the replay config says. Its size
must be the stored frame's, and its AprilTags (cv2.aruco, the 36h11 family the bases read, corners
refined, the centre where the diagonals cross as pupil_apriltags gives it) must lie where the stored
frame has them, within the plan's pixels: per camera, the frames whose re-detected tags all match over
the frames with any re-detected tag is the camera's match rate (the median offset is kept beside it),
and a camera below the plan's bound, or with too few checked frames, is not served (its items are
render errors). The extra check frames the plan drew count towards the rate besides the items' own.
An item whose own frame does not match is a render error too, whatever its camera's rate. A frame or
a clip that fails is that item's error; the others go on.

Rendering changes what an auditor is shown, so it refuses once scored answers exist, unless
--audit-after-answers (logged). It records whether each session's replay config still has the sha256
it had at sampling (the design's values are used either way: the sampling checked them against the
frame sets' times).

Per identity frame: the frame with every person box numbered left to right (blind: thin grey boxes
and numbers only; verify: the display version's pupil and tag source beside the member boxes), and
per box a picture with that box highlighted and the others dimmed (verify: with the version's gaze
ray and class) and a close-up of the head, cut around features.head_box of the person's own keypoints
(else the top of the box) with a margin, upscaled. Per roster crop: the person with their box and a
diamond on the badge read. Per who-speaks window: a clip of every camera in a grid with the group
microphone, without the recordings' metadata, cut by one ffmpeg on two threads at a time.

The images and clips go to artifacts/runtime/audit/<id>/media/<alias>/ (never into a session folder);
render_index.json beside it keeps the frame checks, each head's brightness and the errors, and each
session's view.json gets its items' image names, render status and the boxes the gaze question asks
about after identity (every box a frozen version calls a pupil). Rendering again overwrites.
"""
from __future__ import annotations

import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path

from openmmla.commands.ses import audit as A
from openmmla.commands.ses import code as C

A.L.register_loaded(__name__, __file__)

JPEG_QUALITY = 85
# aruco finds tags whose outline is this share of the image's larger side or more (its default 0.03 misses a
# badge under about 14 px on a 1920 px frame)
MIN_PERIMETER_RATE = 0.01
HEAD_HEIGHT = 256
ROSTER_HEIGHT = 320
CROP_MARGIN = 0.3
TILE = (640, 360)
GREY = (200, 200, 200)
DIMMED = (110, 110, 110)
YELLOW = (0, 230, 255)
CYAN = (255, 220, 0)


# ---- frames ----

class Decoder:
    """one base's video, read as the base read it"""

    def __init__(self, base: dict, design: dict):
        import logging
        from openmmla.streams.video_stream import VideoStream
        # the stream says at length that the file's size and rate are not the defaults it was not given
        logging.getLogger('openmmla.streams.video_stream').setLevel(logging.ERROR)
        self.base = base
        self.sync, self.interval = float(base.get('sync', design['sync'])), float(design['interval'])
        self.rotate = int(design.get('rotate') or 0)
        self.stream = VideoStream('file', file_path=base['path'])
        self.stream.start()
        self.maps = None
        if base.get('fisheye'):
            import cv2
            import numpy as np
            K, D = np.array(base['fisheye']['K']), np.array(base['fisheye']['D'])
            self.maps = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, tuple(design['resolution']), cv2.CV_16SC2)

    def frame(self, k: int):
        """keyframe k as the base processed it, or None past the file's end"""
        import cv2
        read = self.stream.read(start_time=A.video_time(self.sync, self.base['file_start'], self.interval, k))
        if read is None:
            return None
        image = read.data
        if self.maps is not None:
            image = cv2.remap(image, self.maps[0], self.maps[1], interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
        if self.rotate:
            from openmmla.bases.vfa.enums import ROTATIONS
            if self.rotate in ROTATIONS:
                image = cv2.rotate(image, ROTATIONS[self.rotate])
        return image

    def close(self) -> None:
        try:
            self.stream.stop()
        except Exception:
            pass


def tag_centre(points) -> tuple[float, float]:
    """where a tag's diagonals cross (its four corners in order): the centre pupil_apriltags reports, which
    under perspective is not the mean of the corners"""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = [(float(p[0]), float(p[1])) for p in points[:4]]
    # the line from corner 1 to 3 against the line from corner 2 to 4
    d = (x1 - x3) * (y2 - y4) - (y1 - y3) * (x2 - x4)
    if abs(d) < 1e-9:
        return (x1 + x2 + x3 + x4) / 4, (y1 + y2 + y3 + y4) / 4
    a, b = x1 * y3 - y1 * x3, x2 * y4 - y2 * x4
    return (a * (x2 - x4) - (x1 - x3) * b) / d, (a * (y2 - y4) - (y1 - y3) * b) / d


def detect_tags(image) -> dict[str, tuple[float, float]]:
    """the AprilTags (36h11) of an image: id -> centre (tag_centre of its refined corners)"""
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        parameters = cv2.aruco.DetectorParameters()
    else:  # an OpenCV before 4.7
        parameters = cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = getattr(cv2.aruco, 'CORNER_REFINE_APRILTAG', cv2.aruco.CORNER_REFINE_SUBPIX)
    parameters.minMarkerPerimeterRate = MIN_PERIMETER_RATE
    if hasattr(cv2.aruco, 'ArucoDetector'):
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    found = {}
    for corner, tag in zip(corners or [], [] if ids is None else ids.flatten()):
        found[str(int(tag))] = tag_centre(corner.reshape(-1, 2))
    return found


def tag_check(image, stored: dict, px: float) -> dict:
    """whether a decoded frame is the stored one by its AprilTags: 'match' when every stored tag found
    again lies within `px` of where it was stored, 'mismatch' when one lies farther, 'unchecked' when
    none of them is found again; with each tag's offset (found minus stored)"""
    found = detect_tags(image)
    distances, offsets = {}, {}
    for tag, centre in (stored or {}).items():
        if str(tag) in found and centre is not None and len(centre) >= 2:
            fx, fy = found[str(tag)]
            offsets[str(tag)] = (round(fx - float(centre[0]), 2), round(fy - float(centre[1]), 2))
            distances[str(tag)] = math.hypot(fx - float(centre[0]), fy - float(centre[1]))
    if not distances:
        verdict = 'unchecked'
    else:
        verdict = 'match' if max(distances.values()) <= px else 'mismatch'
    return {'verdict': verdict, 'stored': len(stored or {}), 'found_again': len(distances),
            'max_px': round(max(distances.values()), 2) if distances else None, 'offsets': offsets}


def _median(values: list[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    return round(values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2, 2)


# ---- drawing ----

def _scale(image) -> tuple[float, int]:
    height = image.shape[0]
    return max(0.6, height / 900.0), max(1, int(round(height / 540.0)))


def draw_audit(image, boxes: dict, highlight: str | None = None, labels: dict | None = None,
               ray: tuple | None = None) -> list[str]:
    """draw the numbered boxes onto `image` (in place): every box thin and grey with its number, or with
    `highlight` the box of that number thick and yellow and the others dimmed; `labels` (verify mode)
    adds text beside a box's number, `ray` ((x0, y0), (x1, y1), text) a gaze line. Returns the texts
    drawn, so a test can tell a blind picture holds numbers only."""
    import cv2
    font, (scale, thin) = cv2.FONT_HERSHEY_SIMPLEX, _scale(image)
    drawn = []
    for number in sorted(boxes, key=int):
        x1, y1, x2, y2 = (int(round(float(v))) for v in boxes[number][:4])
        if highlight is None:
            colour, width = GREY, thin
        elif number == highlight:
            colour, width = YELLOW, 3 * thin
        else:
            colour, width = DIMMED, thin
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, width)
        text = str(number) + (f" {labels[number]}" if labels and labels.get(number) else '')
        (tw, th), base = cv2.getTextSize(text, font, scale, max(1, thin))
        top = max(0, y1 - th - base - 4)
        cv2.rectangle(image, (x1, top), (x1 + tw + 6, top + th + base + 4), (0, 0, 0), -1)
        cv2.putText(image, text, (x1 + 3, top + th + 2), font, scale, colour if colour != DIMMED else GREY, max(1, thin))
        drawn.append(text)
    if ray is not None:
        (x0, y0), (x1, y1), text = ray
        cv2.line(image, (int(x0), int(y0)), (int(x1), int(y1)), CYAN, 2 * thin)
        cv2.circle(image, (int(x1), int(y1)), 4 * thin, CYAN, -1)
        if text:
            cv2.putText(image, text, (int(x1) + 6, int(y1) - 6), font, scale, CYAN, max(1, thin))
            drawn.append(text)
    return drawn


def _clamped(box, width: int, height: int, margin: float = 0.0) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    dx, dy = margin * (x2 - x1), margin * (y2 - y1)
    x1, y1, x2, y2 = max(0, int(x1 - dx)), max(0, int(y1 - dy)), min(width, int(math.ceil(x2 + dx))), min(height, int(math.ceil(y2 + dy)))
    return (x1, y1, x2, y2) if x2 > x1 + 1 and y2 > y1 + 1 else None


def _resized(crop, height: int):
    import cv2
    scale = height / crop.shape[0]
    return cv2.resize(crop, (max(1, int(round(crop.shape[1] * scale))), height), interpolation=cv2.INTER_CUBIC)


def head_crop(image, person: dict) -> tuple | None:
    """(the close-up of a person's head, upscaled to HEAD_HEIGHT, its mean brightness, the head box it was
    cut around, where the box came from) or None when the head lies outside the frame"""
    import cv2
    found = A.head_crop_box(person)
    if found is None:
        return None
    box, source = found
    height, width = image.shape[:2]
    inner = _clamped(box, width, height)
    outer = _clamped(box, width, height, CROP_MARGIN)
    if inner is None or outer is None:
        return None
    gray = cv2.cvtColor(image[inner[1]:inner[3], inner[0]:inner[2]], cv2.COLOR_BGR2GRAY)
    return (_resized(image[outer[1]:outer[3], outer[0]:outer[2]].copy(), HEAD_HEIGHT), round(float(gray.mean()), 1),
            [round(v, 1) for v in box], source)


def roster_crop(image, bbox, centre):
    """a roster crop: the person with a margin, their box, and a diamond on the badge read"""
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    outer = _clamped(bbox, width, height, CROP_MARGIN)
    if outer is None:
        return None
    picture = image.copy()
    _, thin = _scale(image)
    cv2.rectangle(picture, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), YELLOW, 2 * thin)
    cx, cy, r = float(centre[0]), float(centre[1]), 12 * thin
    diamond = np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], dtype=np.int32)
    cv2.polylines(picture, [diamond], True, CYAN, 2 * thin)
    return _resized(picture[outer[1]:outer[3], outer[0]:outer[2]], ROSTER_HEIGHT)


def _save(path: Path, image) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.jpg')
    if not cv2.imwrite(str(temp), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise OSError(f'cannot write {path.name}')
    os.replace(temp, path)


# ---- clips ----

def clip_command(videos: list[dict], mic: dict | None, start: float, window: float, out: Path) -> list[str]:
    """the ffmpeg command of a who-speaks clip: up to four cameras in a grid of 640 x 360 tiles, the group
    microphone as its sound, exact to the window, without the recordings' metadata, on two threads"""
    videos = videos[:4]
    command = ['ffmpeg', '-v', 'error', '-y']
    for video in videos:
        command += ['-ss', f"{start - video['start_time']:.3f}", '-t', f'{window:.3f}', '-i', video['path']]
    if mic:
        command += ['-ss', f"{start - mic['start_time']:.3f}", '-t', f'{window:.3f}', '-i', mic['path']]
    n, (w, h) = len(videos), TILE
    graph = ''.join(f'[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,'
                    f'setsar=1[v{i}];' for i in range(n))
    if n == 1:
        graph += '[v0]copy[out]'
    elif n == 2:
        graph += '[v0][v1]hstack=inputs=2[out]'
    else:
        if n == 3:
            graph += f'color=c=black:s={w}x{h}:d={window:.3f}[v3];'
        graph += '[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]'
    command += ['-filter_complex', graph, '-map', '[out]']
    if mic:
        command += ['-map', f'{n}:a', '-c:a', 'aac', '-b:a', '96k']
    command += ['-t', f'{window:.3f}', '-map_metadata', '-1', '-threads', '2', '-c:v', 'libx264', '-preset', 'veryfast',
                '-crf', '26', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)]
    return command


def audit_clip(videos: list[dict], mic: dict | None, start: float, window: float, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(f'.{out.stem}.{os.getpid()}.tmp.mp4')
    try:
        subprocess.run(clip_command(videos, mic, start, window, temp), check=True, timeout=300, capture_output=True)
        temp.replace(out)
    finally:
        temp.unlink(missing_ok=True)
    return out


# ---- a session ----

def _versions(artifacts: Path, sid: str, audit_id: str) -> dict:
    out = {}
    for version in A.VERSIONS:
        path = A.version_file(artifacts, sid, audit_id, version)
        if path.exists():
            data = A.read_json(path)
            if not data.get('refused'):
                out[version] = data
    return out


def _verify_labels(design: dict, display: dict | None, item: str) -> dict:
    """verify mode: per member box of the display version, its pupil and where its tag came from"""
    if not display:
        return {}
    letter = {tag: name for name, tag in design['letters'].items()}
    labels = {}
    for number, box in (display['vision'].get(item) or {}).get('boxes', {}).items():
        if box.get('member'):
            source = {'torso': 'read', 'box': 'read', None: 'read'}.get(box.get('match'), box.get('match'))
            labels[number] = f"{letter.get(box['member'], '?')} {source}"
        elif box.get('seat_of'):
            labels[number] = f"{letter.get(box['seat_of'], '?')} seat"
    return labels


def _render_job(image, kind: str, item: dict, name: str | None, check: dict, verdicts, offsets: list, design: dict,
                display: dict | None, verify: bool, media: Path, written, by_item: dict, index: dict) -> str | None:
    """check one decoded frame by its tags and draw its item's pictures: None when done, else why the item
    is not served (its pictures are then deleted by the caller)"""
    frame = item['frame'] if kind != 'check' else item
    if image is None:
        return 'no frame at this time in the video'
    if [image.shape[1], image.shape[0]] != [frame.get('width'), frame.get('height')]:
        verdicts['size'] += 1
        return f"the decoded frame is {image.shape[1]}x{image.shape[0]}, the stored one {frame.get('width')}x{frame.get('height')}"
    result = tag_check(image, frame.get('tags') or {}, float(check['px'])) if frame.get('tags') else {'verdict': 'unchecked'}
    verdicts[result['verdict']] += 1
    offsets.extend((result.get('offsets') or {}).values())
    if kind == 'check':
        return None
    index['items'][name] = {'tag_check': result}
    if result['verdict'] == 'mismatch':
        # this frame is not the one the pipeline analysed, whatever its camera's rate
        return f"its tags are not where they were stored (up to {result['max_px']} px off)"
    if kind == 'roster':
        crop = roster_crop(image, frame['persons'][item['person']]['bbox'], item['centre'])
        if crop is None:
            return 'the person lies outside the frame'
        path = media / f'{name}.jpg'
        _save(path, crop)
        written[name].append(path)
        by_item[name]['image'] = path.name
        return None
    boxes = A.numbered_boxes(frame['persons'])
    rects = {n: b['bbox'] for n, b in boxes.items()}
    ids = image.copy()
    draw_audit(ids, rects, labels=_verify_labels(design, display, name) if verify else None)
    path = media / f'{name}_ids.jpg'
    _save(path, ids)
    written[name].append(path)
    images = {'ids': path.name, 'gaze': {}, 'head': {}}
    heads = {}
    for number, box in boxes.items():
        person = frame['persons'][box['person']]
        picture = image.copy()
        ray = None
        if verify and display:
            values = ((display['vision'].get(name) or {}).get('boxes') or {}).get(number)
            point = ((person.get('gaze') or {}).get('point'))
            head = A.head_crop_box(person)
            if values and point and head:
                (hx1, hy1, hx2, hy2), _ = head
                ray = (((hx1 + hx2) / 2, (hy1 + hy2) / 2), point, values['gaze']['label'])
        draw_audit(picture, rects, highlight=number, ray=ray)
        path = media / f'{name}_gaze_{number}.jpg'
        _save(path, picture)
        written[name].append(path)
        images['gaze'][number] = path.name
        cut = head_crop(image, person)
        if cut is None:
            heads[number] = {'error': 'the head lies outside the frame'}
            continue
        crop, luma, head_box, source = cut
        path = media / f'{name}_head_{number}.jpg'
        _save(path, crop)
        written[name].append(path)
        images['head'][number] = path.name
        heads[number] = {'luma': luma, 'head_box': head_box, 'from': source}
    index['items'][name]['boxes'] = heads
    by_item[name]['images'] = images
    return None


def render_session(artifacts: Path, audit_id: str, plan: dict, entry: dict, check: dict | None = None,
                   cut_clip=audit_clip) -> dict:
    """render one session's items, the frames checked by `check` (the plan's tag_check when None); returns
    the session's part of render_index.json"""
    sid, alias = entry['id'], entry['alias']
    folder = A.session_audit_dir(artifacts, sid, audit_id)
    design, view = A.read_json(folder / A.DESIGN_FILE), A.read_json(folder / A.VIEW_FILE)
    versions = _versions(artifacts, sid, audit_id)
    display = versions.get(design['display_version'])
    verify = plan['mode'] == 'verify'
    media = A.audit_dir(artifacts, audit_id) / A.MEDIA_DIR / alias
    check = check or plan['tag_check']
    index = {'cameras': {}, 'items': {}}
    # the frames each base must decode: (k, kind, entry), in the order the base read them
    wanted: dict[str, list] = defaultdict(list)
    for item in design['vision']:
        wanted[item['base']].append((item['k'], 'vision', item))
    for item in design['roster']:
        wanted[item['base']].append((item['k'], 'roster', item))
    for item in design['checks']:
        wanted[item['base']].append((item['k'], 'check', item))
    by_item = {it['item']: it for it in view['items']}
    written: dict[str, list[Path]] = defaultdict(list)
    errors: dict[str, str] = {}
    for base in design['bases']:
        jobs = sorted(wanted.get(base['id'], []), key=lambda job: (job[0], job[1]))
        verdicts = defaultdict(int)
        offsets: list[tuple[float, float]] = []
        decoder = None
        try:
            decoder = Decoder(base, design)
        except Exception as error:  # a video that cannot be opened: every item of the camera fails
            for _, kind, item in jobs:
                if kind != 'check':
                    errors[item['item']] = f'the video cannot be opened ({type(error).__name__})'
            index['cameras'][str(base['index'])] = {'served': False, 'error': f'cannot open ({type(error).__name__})'}
            continue
        cache: dict[int, object] = {}
        try:
            for k, kind, item in jobs:
                name = item.get('item')
                try:
                    if k not in cache:
                        cache.clear()
                        cache[k] = decoder.frame(k)
                    image = cache[k]
                    result = _render_job(image, kind, item, name, check, verdicts, offsets, design, display, verify,
                                         media, written, by_item, index)
                except Exception as error:  # one frame that fails is its item's error; the others go on
                    if name:
                        errors[name] = f'the frame cannot be drawn ({type(error).__name__})'
                        for path in written.pop(name, []):
                            path.unlink(missing_ok=True)
                    verdicts['failed'] += 1
                    continue
                if result and name:
                    errors[name] = result
                    for path in written.pop(name, []):
                        path.unlink(missing_ok=True)
        finally:
            decoder.close()
        checked = verdicts['match'] + verdicts['mismatch']
        rate = verdicts['match'] / checked if checked else None
        served = checked >= int(check['min_frames']) and rate is not None and rate >= float(check['min_rate'])
        index['cameras'][str(base['index'])] = {'checked': checked, 'match': verdicts['match'],
                                                'mismatch': verdicts['mismatch'], 'unchecked': verdicts['unchecked'],
                                                'size_mismatch': verdicts['size'], 'failed': verdicts['failed'],
                                                'rate': None if rate is None else round(rate, 4), 'served': served,
                                                'median_offset_px': [_median([o[0] for o in offsets]),
                                                                     _median([o[1] for o in offsets])]}
        if not served:
            why = 'the frames could not be checked by their tags' if rate is None or checked < int(check['min_frames']) \
                else f'only {rate:.0%} of the checked frames show their tags where they were stored'
            for _, kind, item in jobs:
                if kind != 'check':
                    errors.setdefault(item['item'], f'camera not served: {why}')
                    for path in written.pop(item['item'], []):
                        path.unlink(missing_ok=True)
    # the gaze question after identity: every box a frozen version calls a pupil
    asked: dict[str, set] = defaultdict(set)
    for data in versions.values():
        for name, values in data.get('vision', {}).items():
            for number, box in (values.get('boxes') or {}).items():
                if box.get('member'):
                    asked[name].add(number)
    for item in design['vision']:
        name = item['item']
        by_item[name]['ask_gaze'] = sorted(asked.get(name, ()), key=int)
        by_item[name]['versions'] = sorted(versions)
    for item in design['speech']:
        name = item['item']
        if not design.get('mic'):
            errors[name] = 'no group microphone'
            continue
        path = media / f'{name}.mp4'
        try:
            cut_clip(design['videos'], design['mic'], float(item['window_start']),
                     float(item['window_end']) - float(item['window_start']), path)
            by_item[name]['clip'] = path.name
        except Exception as error:  # one clip that fails is its window's error; the others go on
            errors[name] = f'the clip cannot be cut ({type(error).__name__})'
    for item in view['items']:
        item['render'] = f"error: {errors[item['item']]}" if item['item'] in errors else 'ok'
        if item['item'] in errors:
            item.pop('images', None)
            item.pop('image', None)
            item.pop('clip', None)
    A.write_json(folder / A.VIEW_FILE, view)
    index['errors'] = errors
    return index


def _ffmpeg_build() -> str | None:
    import cv2
    lines = [line.strip() for line in cv2.getBuildInformation().splitlines() if 'FFMPEG' in line.upper() or 'avcodec' in line]
    return '; '.join(lines[:4]) or None


def replay_state(design: dict) -> str:
    """whether the session's replay config is still the file the sampling checked (the render uses the
    design's sync time, interval and files either way)"""
    recorded = design.get('replay_config') or {}
    if not recorded.get('path'):
        return 'not recorded'
    now = A.L.file_sha256(recorded['path'])
    if now is None:
        return 'gone'
    return 'unchanged' if now == recorded.get('sha256') else 'changed since the sampling'


def cmd_render(args, argv) -> int:
    import cv2
    artifacts, audit_id = A._artifacts(args), args.audit_render
    plan = A.load_plan(artifacts, audit_id)
    answered = A.after_answers(artifacts, plan, args, 'rendering again')
    # the frame check the plan declared, unless a flag says otherwise now (recorded)
    check = {'px': args.audit_tag_px if args.audit_tag_px is not None else plan['tag_check']['px'],
             'min_rate': args.audit_min_tag_match if args.audit_min_tag_match is not None else plan['tag_check']['min_rate'],
             'min_frames': args.audit_min_tag_frames if args.audit_min_tag_frames is not None else plan['tag_check']['min_frames']}
    index = {'audit_id': audit_id, 'rendered_at': C.now_utc(), 'mode': plan['mode'], 'cv2': cv2.__version__,
             'ffmpeg': _ffmpeg_build(), 'tag_check': check, 'tag_check_as_planned': check == plan['tag_check'],
             'after_answers': answered, 'sessions': {}}
    failed, views = 0, []
    for entry in plan['sessions']:
        folder = A.session_audit_dir(artifacts, entry['id'], audit_id)
        try:
            replay = replay_state(A.read_json(folder / A.DESIGN_FILE))
            part = render_session(artifacts, audit_id, plan, entry, check)
        except Exception as error:  # one session that fails is reported; the others go on
            index['sessions'][entry['alias']] = {'failed': f'{type(error).__name__}: {str(error)[:300]}'}
            print(f"{entry['alias']}: not rendered ({type(error).__name__}: {error})")
            failed += 1
            continue
        part['replay_config'] = replay
        index['sessions'][entry['alias']] = part
        views.append(folder / A.VIEW_FILE)
        refused = [c for c, v in part['cameras'].items() if not v.get('served')]
        failed += len(part['errors'])
        print(f"{entry['alias']}: {len(part['items'])} frames drawn, {len(part['errors'])} errors"
              + (f"; camera(s) {', '.join(refused)} not served (frame check)" if refused else '')
              + ('' if replay == 'unchanged' else f'; its replay config is {replay} (the sampled values were used)'))
    A.write_json(A.audit_dir(artifacts, audit_id) / A.RENDER_INDEX, index)
    A._event(artifacts, audit_id, 'audit-render', cv2=cv2.__version__, errors=failed, tag_check=check,
             after_answers=answered, files=A.file_hashes(artifacts, views),
             render_index_sha256=A.L.file_sha256(A.audit_dir(artifacts, audit_id) / A.RENDER_INDEX), argv=list(argv))
    return 0
