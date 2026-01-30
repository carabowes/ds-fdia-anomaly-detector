import numpy as np

# Alarm segmentation utilities

def extract_alarm_segments(alarm_mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Returns list of (start, end) indices for contiguous alarm regions.
    End is exclusive.
    """
    segments = []
    in_segment = False
    start = None

    for t, v in enumerate(alarm_mask):
        if v and not in_segment:
            in_segment = True
            start = t
        elif not v and in_segment:
            segments.append((start, t))
            in_segment = False

    if in_segment:
        segments.append((start, len(alarm_mask)))

    return segments


def overlaps(seg, episode):
    s1, e1 = seg
    s2, e2 = episode
    return max(s1, s2) < min(e1, e2)

# False incident metrics

def compute_false_incident_rate(
    *,
    alarm_segments,
    attack_episodes,
    T,
    normalisation=500
):
    false_incidents = 0

    for seg in alarm_segments:
        if not any(overlaps(seg, ep) for ep in attack_episodes):
            false_incidents += 1

    rate = false_incidents * (normalisation / T)

    return {
        "false_incidents": false_incidents,
        "false_incidents_per_500": rate,
    }

def compute_false_incident_gaps(false_segments):
    """
    Median gap between false incident segments.
    """
    if len(false_segments) < 2:
        return None

    gaps = []
    for (s1, e1), (s2, _) in zip(false_segments[:-1], false_segments[1:]):
        gaps.append(s2 - e1)

    return float(np.median(gaps))

# Episode-level mitigation metrics

def evaluate_episode_detection(
    *,
    attack_episodes,
    alarm_segments,
):
    """
    For each attack episode, determine if it was detected and TTFD.
    """
    results = []

    for ep_start, ep_end in attack_episodes:
        detections = [
            seg for seg in alarm_segments
            if overlaps(seg, (ep_start, ep_end))
        ]

        if not detections:
            results.append({
                "detected": False,
                "ttfd": None,
            })
        else:
            first_seg = min(detections, key=lambda s: s[0])
            ttfd = max(0, first_seg[0] - ep_start)

            results.append({
                "detected": True,
                "ttfd": ttfd,
            })

    return results


def summarise_episode_detection(episode_results):
    """
    Summarise episode-level mitigation performance.
    """
    num_episodes = len(episode_results)
    detected = [r for r in episode_results if r["detected"]]

    num_detected = len(detected)
    detection_rate = num_detected / num_episodes if num_episodes > 0 else 0.0

    ttfds = [r["ttfd"] for r in detected if r["ttfd"] is not None]
    median_ttfd = float(np.median(ttfds)) if ttfds else None

    return {
        "num_episodes": num_episodes,
        "num_detected": num_detected,
        "detection_rate": detection_rate,
        "median_ttfd": median_ttfd,
    }