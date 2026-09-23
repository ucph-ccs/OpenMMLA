"""The transformation matrices between the cameras of a session, from its own recordings, and a
check of given ones. Live, IPS Camera Sync turns every moment two cameras see the same tag into
one pose-to-pose transform (openmmla.bases.ips.transform.direct_transform_matrices) and averages
them. Offline, a whole session's paired sightings are at hand: the tag *positions* the two
cameras report are fitted with one rigid transform (Kabsch), which holds steady where a small
tag's orientation, and so a pose-to-pose transform, does not; the pose-to-pose average is kept
next to it for comparison. Given matrices (Marie's, an earlier session's) are scored by their
residuals on the same pairs, and, for a camera with few pairs, on near-simultaneous sightings
(`near_pairs`, the frames around them found by `near_stamps`) and on its sightings shared with a
third camera already placed (`relayed_pairs`).

Detection runs the way the IPS base does it (pupil-apriltags, the camera's intrinsics, the tag
size; weak detections and ids past the badges left out), on the frames nearest the sampled
stamps of each video. OpenCV and pupil-apriltags are imported only by `observe` and
`tag_detector`, so the fitting can be tested without them."""
from __future__ import annotations

import bisect

import numpy as np

from .transform import average_rotation_matrices, direct_transform_matrices, distance_between_rotations

MAX_TAG_ID = 12  # the badge ids IPS reads (IPSBase.max_badge_id)
MIN_DECISION_MARGIN = 10.0  # a weaker detection is not a badge
ROTATIONS = {90: 0, 180: 1, 270: 2}  # cv2.ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE


def tag_detector(camera_params, tag_size: float, families: str = 'tag36h11'):
    """detect(frame) -> {tag id: (R, t)}: the tags of a BGR frame with their pose in the camera's
    frame, as the IPS base finds them."""
    import cv2
    from pupil_apriltags import Detector
    detector = Detector(families=families, nthreads=4)

    def detect(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        poses = {}
        for tag in detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size):
            if tag.decision_margin < MIN_DECISION_MARGIN or int(tag.tag_id) > MAX_TAG_ID:
                continue
            poses[int(tag.tag_id)] = (np.asarray(tag.pose_R, dtype=float), np.asarray(tag.pose_t, dtype=float).reshape(3))
        return poses
    return detect


def observe(video: str, start_time: float, stamps, detect, rotate: int = 0, progress=None) -> dict:
    """{stamp: {tag: (R, t)}} of one video at the given stamps (unix seconds): the frame nearest
    each stamp is read (seeking), rotated as the base would, and `detect` gives its tags; stamps
    before the file's start are skipped, and the video's end ends the reading."""
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError(f"cannot open {video}")
    out = {}
    try:
        for i, stamp in enumerate(stamps):
            offset = float(stamp) - float(start_time)
            if offset < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, offset * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            if rotate in ROTATIONS:
                frame = cv2.rotate(frame, ROTATIONS[rotate])
            poses = detect(frame)
            if poses:
                out[float(stamp)] = poses
            if progress and i % 100 == 0:
                progress(i, len(out))
    finally:
        cap.release()
    return out


def pairs(main_obs: dict, alt_obs: dict) -> list[tuple]:
    """(stamp, tag, p_main, p_alt, R_main, R_alt) for every tag both cameras saw at one stamp."""
    out = []
    for stamp, main_tags in main_obs.items():
        alt_tags = alt_obs.get(stamp)
        if not alt_tags:
            continue
        for tag, (R_m, t_m) in main_tags.items():
            if tag in alt_tags:
                R_a, t_a = alt_tags[tag]
                out.append((stamp, tag, np.asarray(t_m, float).reshape(3), np.asarray(t_a, float).reshape(3),
                            np.asarray(R_m, float), np.asarray(R_a, float)))
    return out


def near_pairs(main_obs: dict, alt_obs: dict, max_gap: float) -> list[tuple]:
    """(stamp, tag, p_main, p_alt, R_main, R_alt, gap) for every tag the other camera saw, with the
    main camera's sighting of the same tag nearest in time when it lies at most `max_gap` seconds
    away: near-simultaneous sightings, the strict pairs (gap 0) among them. A badge moves little in
    a fifth of a second, so they are good enough to check a transform, not to fit one."""
    by_tag: dict = {}
    for stamp in sorted(main_obs):
        for tag in main_obs[stamp]:
            by_tag.setdefault(tag, []).append(stamp)
    out = []
    for stamp in sorted(alt_obs):
        for tag, (R_a, t_a) in alt_obs[stamp].items():
            stamps = by_tag.get(tag)
            if not stamps:
                continue
            i = bisect.bisect_left(stamps, stamp)
            nearest = min(stamps[max(i - 1, 0):i + 1], key=lambda s: abs(s - stamp))
            gap = abs(nearest - stamp)
            if gap > max_gap + 1e-6:
                continue
            R_m, t_m = main_obs[nearest][tag]
            out.append((stamp, tag, np.asarray(t_m, float).reshape(3), np.asarray(t_a, float).reshape(3),
                        np.asarray(R_m, float), np.asarray(R_a, float), round(gap, 3)))
    return out


def relayed_pairs(via_obs: dict, alt_obs: dict, R, T) -> list[tuple]:
    """(stamp, tag, p_main, p_alt, R_main, R_alt, 0.0) for every tag a camera saw at the same stamp as a
    third camera that is already placed: the third camera's sighting taken into the main camera's
    frame with its transform (R, T) stands in for the main camera's. They check a transform where
    the main camera itself saw no tag with the camera."""
    R, T = np.asarray(R, float), np.asarray(T, float).reshape(3)
    return [(stamp, tag, R @ p_v + T, p_a, R @ R_v, R_a, 0.0) for stamp, tag, p_v, p_a, R_v, R_a in pairs(via_obs, alt_obs)]


def near_stamps(main_obs: dict, alt_obs: dict, step: float, window: float) -> list[float]:
    """the extra stamps worth reading for near-simultaneous sightings: every `window / 2` seconds
    within `window` of each sampled stamp where one camera saw a tag that the other saw one sampled
    step (`step` seconds) before or after, but not at that stamp; the sampled stamps themselves are
    left out."""
    def seen(b: dict, keys: list, stamp: float, tag) -> bool:
        i = bisect.bisect_left(keys, stamp - 1e-3)
        return i < len(keys) and abs(keys[i] - stamp) <= 1e-3 and tag in b[keys[i]]

    def anchors(a: dict, b: dict) -> set:
        keys = sorted(b)
        out = set()
        for stamp, tags in a.items():
            for tag in tags:
                if not seen(b, keys, stamp, tag) and (seen(b, keys, stamp - step, tag) or seen(b, keys, stamp + step, tag)):
                    out.add(stamp)
        return out
    fine = window / 2.0
    extra = set()
    for stamp in anchors(main_obs, alt_obs) | anchors(alt_obs, main_obs):
        for k in (-2, -1, 1, 2):
            extra.add(round(stamp + k * fine, 6))
    return sorted(extra - set(main_obs) - set(alt_obs))


def kabsch(p_alt, p_main) -> tuple[np.ndarray, np.ndarray]:
    """R, T with p_main = R p_alt + T in the least-squares sense over matching rows."""
    p_alt, p_main = np.asarray(p_alt, float), np.asarray(p_main, float)
    c_alt, c_main = p_alt.mean(axis=0), p_main.mean(axis=0)
    H = (p_alt - c_alt).T @ (p_main - c_main)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, c_main - R @ c_alt


def residuals(R, T, p_alt, p_main) -> np.ndarray:
    """|R p_alt + T - p_main| per pair, in metres."""
    return np.linalg.norm(np.asarray(p_alt, float) @ np.asarray(R, float).T + np.asarray(T, float).reshape(3) - np.asarray(p_main, float), axis=1)


def stats(values) -> dict | None:
    v = np.asarray(values, float)
    if v.size == 0:
        return None
    return {'median': round(float(np.median(v)), 3), 'p90': round(float(np.percentile(v, 90)), 3),
            'max': round(float(v.max()), 3), 'rms': round(float(np.sqrt(np.mean(v ** 2))), 3)}


def fit(pair_list: list[tuple], max_residual: float = 0.15, rounds: int = 3) -> tuple[np.ndarray, np.ndarray, dict]:
    """the transform of a camera into the main camera's frame from its paired sightings: a
    Kabsch fit of the positions, refitted without the pairs whose residual is over
    max(max_residual, 3 x the median). Returns (R, T, report): the residuals of the inliers and
    of all pairs, the stamps and tags behind them, and how far the pose-to-pose average (what
    Camera Sync would give from the same inliers) lies from the fit."""
    if len(pair_list) < 3:
        raise ValueError(f"only {len(pair_list)} paired sightings; at least 3 are needed")
    p_main = np.array([p[2] for p in pair_list])
    p_alt = np.array([p[3] for p in pair_list])
    keep = np.ones(len(pair_list), dtype=bool)
    for _ in range(rounds):
        R, T = kabsch(p_alt[keep], p_main[keep])
        res = residuals(R, T, p_alt, p_main)
        limit = max(max_residual, 3.0 * float(np.median(res[keep])))
        new_keep = res <= limit
        if new_keep.sum() < 3 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    R, T = kabsch(p_alt[keep], p_main[keep])
    res = residuals(R, T, p_alt, p_main)
    Rs, Ts = [], []
    for kept, (stamp, tag, pm, pa, Rm, Ra) in zip(keep, pair_list):
        if kept:
            r, t = direct_transform_matrices(Rm, pm.reshape(3, 1), Ra, pa.reshape(3, 1))
            Rs.append(r)
            Ts.append(np.asarray(t, float).reshape(3))
    R_pose = np.asarray(average_rotation_matrices(np.array(Rs)))
    T_pose = np.mean(Ts, axis=0)
    report = {'pairs': len(pair_list), 'inliers': int(keep.sum()), 'stamps': len({p[0] for p in pair_list}),
              'tags': sorted({int(p[1]) for p in pair_list}),
              'residual_m': stats(res[keep]), 'residual_all_m': stats(res),
              'pose_average': {'rotation_difference_deg': round(distance_between_rotations(R, R_pose), 2),
                               'translation_difference_m': round(float(np.linalg.norm(T - T_pose)), 3)}}
    return R, T, report


def verify(R, T, pair_list: list[tuple], fitted=None) -> dict:
    """how given matrices hold on the paired sightings: their residuals, the share within 0.15 m,
    and the difference to the fitted ones when given."""
    p_main = np.array([p[2] for p in pair_list])
    p_alt = np.array([p[3] for p in pair_list])
    res = residuals(R, T, p_alt, p_main)
    out = {'pairs': len(pair_list), 'residual_m': stats(res), 'within_0.15_m': round(float((res <= 0.15).mean()), 3)}
    if fitted is not None:
        R_f, T_f = fitted
        out['difference_to_fit'] = {'rotation_deg': round(distance_between_rotations(np.asarray(R, float), np.asarray(R_f, float)), 2),
                                    'translation_m': round(float(np.linalg.norm(np.asarray(T, float).reshape(3) - np.asarray(T_f, float).reshape(3))), 3)}
    return out


def calibrate(observations: dict, main: str, given: dict | None = None) -> tuple[dict, dict]:
    """{camera: {stamp: {tag: (R, t)}}} -> (the matrices of every other camera into the main
    camera's frame, as transformation_matrices_<main>.json holds them, and a report per camera:
    the fit, and how the given matrices of that camera hold, when there are any)."""
    matrices, report = {}, {'main': main, 'cameras': {}}
    for alt, obs in observations.items():
        if alt == main:
            continue
        pair_list = pairs(observations.get(main, {}), obs)
        entry: dict = {'pairs': len(pair_list)}
        fitted = None
        if len(pair_list) >= 3:
            R, T, fit_report = fit(pair_list)
            matrices[alt] = {'R': R.tolist(), 'T': T.reshape(3, 1).tolist()}
            entry.update(fit_report)
            fitted = (R, T)
        else:
            entry['problem'] = 'fewer than 3 paired sightings: the cameras did not see a tag at the same moment'
        if given and alt in given and pair_list:
            entry['given'] = verify(given[alt]['R'], given[alt]['T'], pair_list, fitted=fitted)
        report['cameras'][alt] = entry
    return matrices, report
