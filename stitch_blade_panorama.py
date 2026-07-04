import argparse
import os
import re

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS


def get_exif_data(img_path):
    try:
        pil_img = Image.open(img_path)
        exif = pil_img._getexif()
    except Exception:
        return None
    if not exif:
        return None

    data = {}
    for k, v in exif.items():
        tag = TAGS.get(k, k)
        if tag == "GPSInfo":
            gps = {}
            for gk, gv in v.items():
                gps[GPSTAGS.get(gk, gk)] = gv
            if "GPSLatitude" in gps:
                def to_dec(dms, ref):
                    d = float(dms[0])
                    m = float(dms[1])
                    s = float(dms[2])
                    dec = d + m / 60.0 + s / 3600.0
                    if str(ref) in ("S", "W"):
                        dec = -dec
                    return dec

                data["lat"] = to_dec(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
                data["lon"] = to_dec(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
                data["alt"] = float(gps["GPSAltitude"]) if "GPSAltitude" in gps else 0.0
        elif tag == "FocalLengthIn35mmFilm":
            data["focal_35mm"] = float(v)

    try:
        with open(img_path, "rb") as f:
            raw = f.read()
        for xmp_name, data_name in (
            (b"GimbalYawDegree", "gimbal_yaw"),
            (b"GimbalPitchDegree", "gimbal_pitch"),
            (b"GimbalRollDegree", "gimbal_roll"),
            (b"FlightYawDegree", "flight_yaw"),
            (b"RelativeAltitude", "relative_alt"),
        ):
            m = re.search(rb"drone-dji:" + xmp_name + rb'="([+-]?[0-9.]+)"', raw)
            if m:
                data[data_name] = float(m.group(1))
    except Exception:
        pass
    return data


def gps_to_enu(lat, lon, alt, origin_lat, origin_lon, origin_alt):
    lat_m = 111320.0
    lon_m = 111320.0 * np.cos(np.radians(origin_lat))
    return (lon - origin_lon) * lon_m, (lat - origin_lat) * lat_m, alt - origin_alt


def estimate_gsd(alt, focal_35mm=70, img_width=4032):
    fov_h = 2.0 * np.degrees(np.arctan(36.0 / (2.0 * focal_35mm)))
    ground_width = 2.0 * alt * np.tan(np.radians(fov_h / 2.0))
    return ground_width / img_width


def get_camera_yaw(meta):
    return meta.get("gimbal_yaw", meta.get("flight_yaw"))


def circular_mean_deg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    radians = np.radians(vals)
    return float(np.degrees(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))))


def mean_deg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def project_enu_to_camera(de, dn, du, yaw_deg, pitch_deg=None):
    yaw_rad = np.radians(yaw_deg)
    right_m = de * np.cos(yaw_rad) - dn * np.sin(yaw_rad)
    forward_m = de * np.sin(yaw_rad) + dn * np.cos(yaw_rad)

    if pitch_deg is None:
        return right_m, forward_m

    pitch_down_rad = np.radians(-pitch_deg)
    vertical_m = forward_m * np.sin(pitch_down_rad) + du * np.cos(pitch_down_rad)
    return right_m, vertical_m


def projected_pair_displacement(enu, gps_data, idx, fallback_yaw, fallback_pitch, use_camera_projection):
    de = enu[idx][0] - enu[idx - 1][0]
    dn = enu[idx][1] - enu[idx - 1][1]
    du = enu[idx][2] - enu[idx - 1][2]

    if not use_camera_projection:
        return de, dn

    pair_yaw = circular_mean_deg([get_camera_yaw(gps_data[idx - 1]), get_camera_yaw(gps_data[idx])])
    pair_pitch = mean_deg([gps_data[idx - 1].get("gimbal_pitch"), gps_data[idx].get("gimbal_pitch")])
    return project_enu_to_camera(
        de,
        dn,
        du,
        fallback_yaw if pair_yaw is None else pair_yaw,
        fallback_pitch if pair_pitch is None else pair_pitch,
    )


def mat_mul3x3(a, b):
    return a @ b


def transform_points(mat, pts):
    pts_h = np.concatenate(
        [pts.reshape(-1, 2).astype(np.float64), np.ones((pts.shape[0], 1))],
        axis=1,
    )
    out = pts_h @ mat.T
    out[:, 0] /= np.maximum(np.abs(out[:, 2]), 1e-12) * np.sign(out[:, 2])
    out[:, 1] /= np.maximum(np.abs(out[:, 2]), 1e-12) * np.sign(out[:, 2])
    return out[:, :2].reshape(-1, 1, 2)


def extract_blade_mask(img, low_sat_percentile=12):
    h, w = img.shape[:2]
    img_area = h * w

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1].astype(np.float32)
    v_channel = hsv[:, :, 2]
    gaussian = cv2.GaussianBlur(s_channel, (15, 15), 0)

    sat_threshold = np.percentile(gaussian, low_sat_percentile)
    sat_threshold = min(max(sat_threshold, 8), 30)

    b, g, r = cv2.split(img)
    rgb_diff = np.maximum(
        np.maximum(
            np.abs(r.astype(np.float32) - g.astype(np.float32)),
            np.abs(g.astype(np.float32) - b.astype(np.float32)),
        ),
        np.abs(b.astype(np.float32) - r.astype(np.float32)),
    )

    mask = ((gaussian < sat_threshold) & (v_channel > 40) & (rgb_diff < 50)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask), np.zeros_like(mask, dtype=np.float32), 0.0

    scored = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.0005:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        center_dist = np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
        dist_score = max(0.0, 1.0 - center_dist / (max(w, h) * 0.75))

        mask_single = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_single, [c], -1, 255, -1)

        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        aspect = max(rw, rh) / max(min(rw, rh), 1) if rw > 0 and rh > 0 else 1.0
        elongation = min(aspect / 8.0, 3.0)

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        region_hsv = hsv[mask_single > 0]
        mean_s = np.mean(region_hsv[..., 1]) if len(region_hsv) > 0 else 255
        mean_v = np.mean(region_hsv[..., 2]) if len(region_hsv) > 0 else 0

        region_rgb = img[mask_single > 0]
        if region_rgb.ndim == 1:
            region_rgb = region_rgb.reshape(-1, 3)
        if region_rgb.ndim == 2 and region_rgb.shape[1] >= 3 and len(region_rgb) > 1:
            rgb_spread = np.mean([np.std(region_rgb[:, ch].astype(np.float32)) for ch in range(3)])
            uniformity = 1.0 / (1.0 + rgb_spread / 15.0)
        else:
            uniformity = 0.0

        area_rel = min(area / (img_area * 0.15), 2.0)
        low_s = max(0.0, 1.0 - mean_s / 60.0)
        bright = min(mean_v / 200.0, 1.0)

        score = (
            area_rel * 2.0
            + dist_score * 3.0
            + elongation * 2.5
            + solidity * 1.5
            + uniformity * 2.0
            + low_s * 2.5
            + bright
        )
        scored.append((score, c, area))

    if not scored:
        return np.zeros_like(mask), np.zeros_like(mask, dtype=np.float32), 0.0

    scored.sort(key=lambda x: x[0], reverse=True)
    blade_mask = np.zeros_like(mask)
    _, best_contour, total_area = scored[0]
    cv2.drawContours(blade_mask, [best_contour], -1, 255, -1)

    if total_area < img_area * 0.002:
        return np.zeros_like(mask), np.zeros_like(mask, dtype=np.float32), 0.0

    blade_mask = cv2.dilate(
        blade_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=2,
    )
    dist = cv2.distanceTransform(blade_mask, cv2.DIST_L2, 5)
    alpha = np.clip(dist / 18.0, 0, 1).astype(np.float32)
    return blade_mask, alpha, total_area / img_area


def match_rotation(
    img1,
    img2,
    gps_dx,
    gps_dy,
    use_gps=True,
    visual_refine=True,
    max_visual_shift=40.0,
    visual_weight=0.35,
):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(des1) < 30 or len(des2) < 30:
        h = np.eye(3, dtype=np.float64)
        h[0, 2] = gps_dx
        h[1, 2] = gps_dy
        return h, "few", 0.0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    tol = max(abs(gps_dx) * 0.5 + 200, 250) if use_gps else 1e9

    good = []
    dxs = []
    dys = []
    for m in matches[:350]:
        p1 = kp1[m.queryIdx].pt
        p2 = kp2[m.trainIdx].pt
        dx_obs = p1[0] - p2[0]
        dy_obs = p1[1] - p2[1]
        if abs(dx_obs - gps_dx) < tol and abs(dy_obs - gps_dy) < tol:
            good.append(m)
            dxs.append(dx_obs)
            dys.append(dy_obs)

    n_good = len(good)
    if n_good < 10:
        good = matches[: min(220, len(matches))]
        dxs = [kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0] for m in good]
        dys = [kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1] for m in good]
    if len(good) < 6:
        h = np.eye(3, dtype=np.float64)
        h[0, 2] = gps_dx
        h[1, 2] = gps_dy
        return h, f"few({len(good)})", 0.0, 0.0

    feat_dx = float(np.median(dxs))
    feat_dy = float(np.median(dys))
    refined_dx = gps_dx
    refined_dy = gps_dy
    if visual_refine:
        shift_x = np.clip(feat_dx - gps_dx, -max_visual_shift, max_visual_shift)
        shift_y = np.clip(feat_dy - gps_dy, -max_visual_shift, max_visual_shift)
        refined_dx = gps_dx + shift_x * visual_weight
        refined_dy = gps_dy + shift_y * visual_weight

    angles = [np.arctan2(dy - refined_dy, dx - refined_dx) for dx, dy in zip(dxs, dys)]
    theta = float(np.median(angles)) if angles else 0.0
    theta_deg = np.degrees(theta)
    if abs(theta_deg) > 15:
        theta = 0.0
        theta_deg = 0.0

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    h = np.eye(3, dtype=np.float64)
    h[0, 0] = cos_t
    h[0, 1] = -sin_t
    h[1, 0] = sin_t
    h[1, 1] = cos_t
    h[0, 2] = refined_dx
    h[1, 2] = refined_dy
    return h, f"rot={theta_deg:.1f}deg n={n_good} gps=({gps_dx:.0f},{gps_dy:.0f}) vis=({feat_dx:.0f},{feat_dy:.0f})", feat_dx, feat_dy


def warp_and_blend(canvas, new_img, h_mat):
    h1, w1 = canvas.shape[:2]
    h2, w2 = new_img.shape[:2]
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_c = transform_points(h_mat, corners2)
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    all_c = np.vstack((corners1, warped_c)).reshape(-1, 2)

    xmin = int(np.floor(np.min(all_c[:, 0])))
    ymin = int(np.floor(np.min(all_c[:, 1])))
    xmax = int(np.ceil(np.max(all_c[:, 0])))
    ymax = int(np.ceil(np.max(all_c[:, 1])))

    t_mat = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
    size = (xmax - xmin, ymax - ymin)
    canvas_warped = cv2.warpPerspective(canvas, t_mat, size)
    new_warped = cv2.warpPerspective(new_img, t_mat @ h_mat, size)

    black = np.all(canvas_warped == 0, axis=-1)
    canvas_warped[black] = new_warped[black]
    overlap = (~black) & np.any(new_warped > 0, axis=-1)
    if np.any(overlap):
        seam = cv2.erode(
            overlap.astype(np.uint8) * 255,
            np.ones((25, 25), np.uint8),
            iterations=2,
        ).astype(bool)
        blended = cv2.addWeighted(canvas_warped, 0.5, new_warped, 0.5, 0)
        canvas_warped[seam] = blended[seam]

    gray = cv2.cvtColor(canvas_warped, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero((gray > 0).astype(np.uint8))
    if coords is None:
        return canvas_warped, 0.0, 0.0
    x, y, w, h = cv2.boundingRect(coords)
    return canvas_warped[y : y + h, x : x + w], float(x), float(y)


def warp_and_accumulate(accum, weight, new_img, new_alpha, h_mat):
    h1, w1 = weight.shape[:2]
    h2, w2 = new_img.shape[:2]
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_c = transform_points(h_mat, corners2)
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    all_c = np.vstack((corners1, warped_c)).reshape(-1, 2)

    xmin = int(np.floor(np.min(all_c[:, 0])))
    ymin = int(np.floor(np.min(all_c[:, 1])))
    xmax = int(np.ceil(np.max(all_c[:, 0])))
    ymax = int(np.ceil(np.max(all_c[:, 1])))
    t_mat = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
    size = (xmax - xmin, ymax - ymin)

    accum_warped = cv2.warpPerspective(accum, t_mat, size)
    weight_warped = cv2.warpPerspective(weight, t_mat, size)
    alpha_warped = cv2.warpPerspective(new_alpha, t_mat @ h_mat, size)
    premult = new_img.astype(np.float32) * new_alpha[..., None]
    premult_warped = cv2.warpPerspective(premult, t_mat @ h_mat, size)

    accum_warped += premult_warped
    weight_warped += alpha_warped

    coords = cv2.findNonZero((weight_warped > 0.01).astype(np.uint8))
    if coords is None:
        return accum_warped, weight_warped
    x, y, w, h = cv2.boundingRect(coords)
    return accum_warped[y : y + h, x : x + w], weight_warped[y : y + h, x : x + w]


def composite_to_canvas(images, alphas, transforms, size):
    width, height = size
    accum = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    for idx, (img, alpha, h_mat) in enumerate(zip(images, alphas, transforms)):
        premult = img.astype(np.float32) * alpha[..., None]
        premult_warped = cv2.warpPerspective(premult, h_mat, (width, height))
        alpha_warped = cv2.warpPerspective(alpha.astype(np.float32), h_mat, (width, height))
        accum += premult_warped
        weight += alpha_warped
        print(f"  {idx + 1:02d}/{len(images)}: accumulated")

    coords = cv2.findNonZero((weight > 0.01).astype(np.uint8))
    if coords is None:
        return accum, weight
    x, y, w, h = cv2.boundingRect(coords)
    return accum[y : y + h, x : x + w], weight[y : y + h, x : x + w]


def scale_transform(h_mat, scale_factor):
    if abs(scale_factor - 1.0) < 1e-9:
        return h_mat.copy()
    down = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]], dtype=np.float64)
    up = np.array([[1.0 / scale_factor, 0, 0], [0, 1.0 / scale_factor, 0], [0, 0, 1]], dtype=np.float64)
    return up @ h_mat @ down


def composite_native_to_canvas(input_dir, gps_data, low_ratios, transforms, scale_factor, size, mask_percentile, min_mask_ratio, blend_mode):
    width, height = size
    accum = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    direct = np.zeros((height, width, 3), dtype=np.uint8)

    for idx, (g, h_low) in enumerate(zip(gps_data, transforms)):
        if low_ratios[idx] <= 0:
            print(f"  {idx + 1:02d}/{len(gps_data)}: skipped")
            continue

        img = cv2.imread(os.path.join(input_dir, g["file"]))
        if img is None:
            print(f"  {idx + 1:02d}/{len(gps_data)}: unreadable")
            continue

        _, alpha, ratio = extract_blade_mask(img, mask_percentile)
        if ratio < min_mask_ratio:
            print(f"  {idx + 1:02d}/{len(gps_data)}: skipped native mask={ratio * 100:.1f}%")
            continue

        h_native = scale_transform(h_low, scale_factor)
        alpha_warped = cv2.warpPerspective(alpha.astype(np.float32), h_native, (width, height))

        if blend_mode == "average":
            premult = img.astype(np.float32) * alpha[..., None]
            premult_warped = cv2.warpPerspective(premult, h_native, (width, height))
            accum += premult_warped
            weight += alpha_warped
        else:
            img_warped = cv2.warpPerspective(img, h_native, (width, height), flags=cv2.INTER_LINEAR)
            replace = alpha_warped > weight
            direct[replace] = img_warped[replace]
            weight[replace] = alpha_warped[replace]
        print(f"  {idx + 1:02d}/{len(gps_data)}: accumulated native")

    coords = cv2.findNonZero((weight > 0.01).astype(np.uint8))
    if coords is None:
        return (accum if blend_mode == "average" else direct.astype(np.float32)), weight
    x, y, w, h = cv2.boundingRect(coords)
    if blend_mode == "average":
        return accum[y : y + h, x : x + w], weight[y : y + h, x : x + w]
    return direct[y : y + h, x : x + w].astype(np.float32), weight[y : y + h, x : x + w]


def composite_native_full_frame(input_dir, gps_data, transforms, scale_factor, size):
    width, height = size
    direct = np.zeros((height, width, 3), dtype=np.uint8)
    weight = np.zeros((height, width), dtype=np.float32)

    for idx, (g, h_low) in enumerate(zip(gps_data, transforms)):
        img = cv2.imread(os.path.join(input_dir, g["file"]))
        if img is None:
            print(f"  {idx + 1:02d}/{len(gps_data)}: unreadable")
            continue

        h_native = scale_transform(h_low, scale_factor)
        warped = cv2.warpPerspective(img, h_native, (width, height), flags=cv2.INTER_LINEAR)
        src_mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(src_mask, h_native, (width, height), flags=cv2.INTER_NEAREST)
        valid = warped_mask > 0
        direct[valid] = warped[valid]
        weight[valid] = 1.0
        print(f"  {idx + 1:02d}/{len(gps_data)}: pasted native frame")

    coords = cv2.findNonZero((weight > 0).astype(np.uint8))
    if coords is None:
        return direct.astype(np.float32), weight
    x, y, w, h = cv2.boundingRect(coords)
    return direct[y : y + h, x : x + w].astype(np.float32), weight[y : y + h, x : x + w]


def load_inputs(input_dir):
    files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    )
    gps_data = []
    for f in files:
        exif = get_exif_data(os.path.join(input_dir, f))
        if exif and "lat" in exif:
            exif["file"] = f
            gps_data.append(exif)
    return gps_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input")
    parser.add_argument("--output", default="stitched_blade_panorama.png")
    parser.add_argument("--scale", type=int, default=1500, help="Native-width scale target, 0 keeps full resolution")
    parser.add_argument("--low-res-output", action="store_true", help="Composite the scaled registration images instead of the native originals")
    parser.add_argument("--full-frame", action="store_true", help="Paste whole native input frames by the estimated coordinates, without blade masks or blending")
    parser.add_argument("--blend-mode", choices=("direct", "average"), default="direct", help="Native output compositing mode")
    parser.add_argument("--gsd-mode", choices=("exif", "visual"), default="exif", help="Pixel scale source for GPS displacement")
    parser.add_argument("--gps-projection", choices=("camera", "enu"), default="camera", help="Project GPS displacement through DJI camera yaw/pitch or use raw ENU axes")
    parser.add_argument("--gps-x-scale", type=float, default=1.0, help="Extra multiplier for projected GPS x displacement")
    parser.add_argument("--gps-y-sign", choices=("same", "invert"), default="invert", help="Map projected GPS forward/northing delta to image y with the same sign or inverted sign")
    parser.add_argument("--no-visual-refine", action="store_true", help="Disable visual translation refinement and use GPS displacement directly")
    parser.add_argument("--max-visual-shift", type=float, default=40.0, help="Maximum per-pair visual correction in scaled pixels")
    parser.add_argument("--visual-weight", type=float, default=0.35, help="Blend weight for visual correction, from 0.0 to 1.0")
    parser.add_argument("--mask-percentile", type=float, default=12)
    parser.add_argument("--min-mask-ratio", type=float, default=0.02, help="Skip frames whose blade mask area ratio is below this value")
    parser.add_argument("--save-mask-preview", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, args.input)
    output_path = os.path.join(script_dir, args.output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    gps_data = load_inputs(input_dir)
    if len(gps_data) < 2:
        raise RuntimeError("Need at least two GPS-tagged images")

    print(f"Found {len(gps_data)} GPS-tagged images")

    origin_lat = gps_data[0]["lat"]
    origin_lon = gps_data[0]["lon"]
    origin_alt = gps_data[0]["alt"]
    enu = [
        gps_to_enu(g["lat"], g["lon"], g["alt"], origin_lat, origin_lon, origin_alt)
        for g in gps_data
    ]

    first_img = cv2.imread(os.path.join(input_dir, gps_data[0]["file"]))
    if first_img is None:
        raise RuntimeError(f"Failed to read first image: {gps_data[0]['file']}")
    native_h, native_w = first_img.shape[:2]
    scale_factor = 1.0 if args.scale <= 0 else args.scale / float(native_w)

    print("Loading images and blade masks...")
    images = []
    alphas = []
    masks = []
    ratios = []
    for idx, g in enumerate(gps_data):
        img = cv2.imread(os.path.join(input_dir, g["file"]))
        if img is None:
            continue
        if args.scale > 0:
            h, w = img.shape[:2]
            img = cv2.resize(
                img,
                (int(w * scale_factor), int(h * scale_factor)),
                interpolation=cv2.INTER_AREA,
            )
        mask, alpha, ratio = extract_blade_mask(img, args.mask_percentile)
        if ratio < args.min_mask_ratio:
            mask = np.zeros_like(mask)
            alpha = np.zeros_like(alpha)
            ratio = 0.0
        images.append(img)
        masks.append(mask)
        alphas.append(alpha)
        ratios.append(ratio)
        print(f"  {idx + 1:02d}/{len(gps_data)} {g['file']} blade={ratio * 100:.1f}%")

    n = len(images)
    if n < 2:
        raise RuntimeError("Need at least two readable images")

    if args.save_mask_preview:
        thumbs = []
        for idx, img in enumerate(images):
            overlay = img.copy()
            overlay[masks[idx] > 0] = (
                overlay[masks[idx] > 0].astype(np.float32) * 0.35
                + np.array([0, 0, 255], dtype=np.float32) * 0.65
            ).astype(np.uint8)
            thumb = cv2.resize(overlay, (300, 225), interpolation=cv2.INTER_AREA)
            cv2.putText(
                thumb,
                f"{idx:02d} {ratios[idx] * 100:.1f}%",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            thumbs.append(thumb)
        rows = []
        for i in range(0, len(thumbs), 4):
            row = thumbs[i : i + 4]
            while len(row) < 4:
                row.append(np.zeros_like(thumbs[0]))
            rows.append(cv2.hconcat(row))
        preview = cv2.vconcat(rows)
        preview_path = os.path.splitext(output_path)[0] + "_masks.jpg"
        cv2.imwrite(preview_path, preview)
        print(f"Mask preview: {preview_path}")

    avg_alt = np.mean([g["alt"] for g in gps_data])
    avg_focal = np.mean([g.get("focal_35mm", 70) for g in gps_data])
    exif_gsd = estimate_gsd(avg_alt, avg_focal) / scale_factor

    yaw_values = [get_camera_yaw(g) for g in gps_data if get_camera_yaw(g) is not None]
    pitch_values = [g.get("gimbal_pitch") for g in gps_data if g.get("gimbal_pitch") is not None]
    fallback_yaw = circular_mean_deg(yaw_values)
    fallback_pitch = mean_deg(pitch_values)
    use_camera_projection = args.gps_projection == "camera" and fallback_yaw is not None
    if use_camera_projection:
        pitch_msg = f" pitch={fallback_pitch:.1f} deg" if fallback_pitch is not None else " pitch=none"
        print(f"  GPS projection: camera yaw={fallback_yaw:.1f} deg{pitch_msg}")
    else:
        print("  GPS projection: raw ENU")

    if args.gsd_mode == "visual":
        print("Calibrating GPS pixel scale from visual matches...")
        gsd_samples = []
        for i in range(1, n):
            _, _, fd, fdy = match_rotation(images[i - 1], images[i], 0, 0, use_gps=False)
            if abs(fd) > 10 or abs(fdy) > 10:
                gps_dx, gps_dy = projected_pair_displacement(
                    enu,
                    gps_data,
                    i,
                    fallback_yaw,
                    fallback_pitch,
                    use_camera_projection,
                )
                if abs(gps_dx) > 0.1 and abs(fd) > 10:
                    gsd_samples.append(abs(gps_dx) / abs(fd))
                else:
                    gps_dist = np.hypot(gps_dx, gps_dy)
                    feat_dist = np.hypot(fd, fdy)
                    if gps_dist > 0.1 and feat_dist > 10:
                        gsd_samples.append(gps_dist / feat_dist)

        if len(gsd_samples) >= 5:
            gsd = float(np.median(gsd_samples))
            print(f"  GSD visual median: {gsd * 100:.2f} cm/scaled px from {len(gsd_samples)} samples")
        else:
            gsd = exif_gsd
            print(f"  GSD visual fallback to EXIF: {gsd * 100:.2f} cm/scaled px")
    else:
        gsd = exif_gsd
        print(f"  GSD EXIF/GPS: {gsd * 100:.2f} cm/scaled px")

    y_sign = 1.0 if args.gps_y_sign == "same" else -1.0

    pixel_dx = []
    pixel_dy = []
    for i in range(1, n):
        dx_m, dy_m = projected_pair_displacement(
            enu,
            gps_data,
            i,
            fallback_yaw,
            fallback_pitch,
            use_camera_projection,
        )
        pixel_dx.append(dx_m / gsd * args.gps_x_scale)
        pixel_dy.append(y_sign * dy_m / gsd)
    print(f"  Total GPS displacement: dx={np.sum(pixel_dx):.0f} dy={np.sum(pixel_dy):.0f} px")

    print("Computing pair transforms...")
    image_transforms = [np.eye(3, dtype=np.float64)]
    acc = images[0].copy()
    crop_x = 0.0
    crop_y = 0.0
    for i in range(1, n):
        init_dx = sum(pixel_dx[:i]) - crop_x
        init_dy = sum(pixel_dy[:i]) - crop_y
        prev_h, prev_w = images[i - 1].shape[:2]
        margin = 400
        x_start = max(0, acc.shape[1] - prev_w - margin)
        y_start = max(0, (acc.shape[0] - prev_h) // 2 - margin)
        y_end = min(acc.shape[0], y_start + prev_h + margin * 2)
        roi = acc[y_start:y_end, x_start:]

        h_mat, info, _, _ = match_rotation(
            roi,
            images[i],
            init_dx - x_start,
            init_dy - y_start,
            visual_refine=not args.no_visual_refine,
            max_visual_shift=args.max_visual_shift,
            visual_weight=args.visual_weight,
        )
        h_mat[0, 2] += x_start
        h_mat[1, 2] += y_start

        acc, cx, cy = warp_and_blend(acc, images[i], h_mat)
        crop_t = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
        image_transforms = [crop_t @ t for t in image_transforms]
        image_transforms.append(crop_t @ h_mat)
        crop_x += cx
        crop_y += cy
        print(f"  {i:02d}/{n - 1}: {info} -> {acc.shape[1]}x{acc.shape[0]}")

    print("Compositing masked blade pixels...")
    if args.low_res_output or abs(scale_factor - 1.0) < 1e-9:
        accum, weight = composite_to_canvas(images, alphas, image_transforms, (acc.shape[1], acc.shape[0]))
    else:
        native_size = (
            int(np.ceil(acc.shape[1] / scale_factor)),
            int(np.ceil(acc.shape[0] / scale_factor)),
        )
        print(f"  Native canvas: {native_size[0]}x{native_size[1]}")
        if args.full_frame:
            accum, weight = composite_native_full_frame(input_dir, gps_data, image_transforms, scale_factor, native_size)
        else:
            accum, weight = composite_native_to_canvas(
                input_dir,
                gps_data,
                ratios,
                image_transforms,
                scale_factor,
                native_size,
                args.mask_percentile,
                args.min_mask_ratio,
                args.blend_mode,
            )

    out_rgb = np.zeros_like(accum, dtype=np.uint8)
    valid = weight > 0.01
    if args.blend_mode == "average" or args.low_res_output or abs(scale_factor - 1.0) < 1e-9:
        out_rgb[valid] = np.clip(accum[valid] / weight[valid, None], 0, 255).astype(np.uint8)
    else:
        out_rgb[valid] = np.clip(accum[valid], 0, 255).astype(np.uint8)
    alpha_u8 = np.clip(weight / max(float(weight.max()), 1e-6) * 255, 0, 255).astype(np.uint8)

    if output_path.lower().endswith(".png"):
        bgra = cv2.cvtColor(out_rgb, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha_u8
        ok = cv2.imwrite(output_path, bgra)
    else:
        ok = cv2.imwrite(output_path, out_rgb)
    if not ok:
        raise RuntimeError(f"Failed to write {output_path}")

    print(f"Saved: {output_path}")
    print(f"Size: {out_rgb.shape[1]}x{out_rgb.shape[0]}")
    print(f"File: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
