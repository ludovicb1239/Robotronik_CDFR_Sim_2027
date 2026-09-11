using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Estimates the robot's real position from a lidar scan, starting from an
/// approximate position (typically odometry) and a known field outline.
/// All lengths are millimetres, all angles are degrees.
///
/// Convention: X is forward, 0 deg points along +X, Y is left, 90 deg points
/// along +Y. Angles are counter-clockwise.
///
/// Scoring: each ray scores exp(-d^2 / 2*sigma^2) against the distance d from
/// its hit point to the nearest field wall, and the pose score is the mean over
/// rays. There is no plateau and no cutoff, so the score is smooth everywhere
/// and every ray contributes to the gradient.
///
/// Algorithm: zooming coordinate-descent scan matching. Instead of scoring the
/// whole (dx, dy, da) box, the search sweeps one axis at a time - heading, then
/// X, then Y - because a robot pose error is dominated by a single axis at a
/// time and each axis converges independently around a tight odometry prior.
/// After every sweep the search range shrinks by ZOOM_FACTOR, so the same sweep
/// progressively changes from broad exploration into fine refinement of the
/// peak, and the sub-cell accuracy falls out of the final narrow sweeps rather
/// than a separate parabola fit.
///
/// Cost: a box search samples (2na+1)(2nx+1)(2ny+1) full poses, while one sweep
/// samples only (2na+1)+(2nx+1)+(2ny+1) and that count is fixed by
/// POINTS_PER_SWEEP regardless of how coarse or fine the range is. The sweeps
/// also reuse work the box cannot: the scan is projected once per sweep and each
/// candidate is a cheap translation or rotation of those points.
///
/// The inner loop is kept cheap by three things, in order of impact:
/// 1. distance-to-wall comes from a baked distance field (a bilinear lookup)
///    instead of a walk over every wall segment;
/// 2. each ray's range and bearing are baked once per scan, so projection
///    carries no per-ray trigonometry;
/// 3. the score is a plain Gaussian, which the compiler turns into a single
///    multiply and an Exp.
///
/// Note: scoring every ray matters for correctness, not just cost. Rays that hit
/// obstacles are repeatable features, and dropping them leaves the search with
/// too few constraints, which lets it drift. Obstacle rays are therefore kept.
/// </summary>
public static class PosEstimator
{
    /// <summary>Outline of the playing field, as a closed polygon, in mm.</summary>
    private static readonly Vector2[] FIELD_OUTLINE =
    {
        new Vector2(  1000f, 1500f),
        new Vector2(  1000f,-1500f),
        new Vector2( -1000f,-1500f),
        new Vector2( -1000f, -900f),
        new Vector2(  -545f, -900f),
        new Vector2(  -545f,  900f),
        new Vector2( -1000f,  900f),
        new Vector2( -1000f, 1500f),
    };

    /// <summary>Falloff of the score with distance from a wall, in mm.</summary>
    private const float SCORE_SIGMA_MM = 20f;

    // --- Distance field ----------------------------------------------------
    /// <summary>
    /// Cell size of the baked distance-to-nearest-wall grid, in mm. The field is
    /// sampled bilinearly, so the error is a small fraction of a cell; 2 mm
    /// keeps it far below the lidar's own precision while staying cache friendly.
    /// </summary>
    private const float FIELD_CELL_MM = 2f;

    /// <summary>Half-extent of the baked grid, covering the field plus a margin, in mm.</summary>
    private const float FIELD_HALF_MM = 2000f;

    /// <summary>Number of cells along each axis of the distance field.</summary>
    private const int FIELD_SIZE = 2001; // 2*FIELD_HALF_MM / FIELD_CELL_MM + 1

    /// <summary>
    /// Distance from every grid node to the nearest wall, in tenths of a
    /// millimetre. Stored as short rather than float to halve the working set:
    /// 0.1 mm quantisation is far below the 0.5 mm target and the field is
    /// sampled millions of times, so cache residency matters.
    /// </summary>
    private static short[] distance_field;

    /// <summary>Scale from the stored short back to millimetres.</summary>
    private const float FIELD_UNIT_MM = 0.1f;

    /// <summary>What the distance field is evaluated to at least, in mm.</summary>
    private const float FIELD_MAX_DISTANCE_MM = 500f;

    // Per-ray invariants for the current scan, baked by PrepareScan.
    private static float[] scan_cos;
    private static float[] scan_sin;
    private static float[] scan_range_mm;

    // --- Search schedule ---------------------------------------------------
    /// <summary>Half-width of the initial search box around the approximate pose, in mm.</summary>
    private const float RANGE_MM = 200f;

    /// <summary>Half-width of the initial search box around the approximate heading, in degrees.</summary>
    private const float ANGLE_RANGE_DEG = 20f;

    /// <summary>Total number of coordinate-descent sweeps performed per estimate.</summary>
    private const int TOTAL_STEPS = 12;

    /// <summary>
    /// Sample points taken on each side of the current best, per sweep. A sweep
    /// therefore evaluates 2*POINTS_PER_SWEEP+1 candidates per axis.
    /// </summary>
    private const int POINTS_PER_SWEEP = 20;

    /// <summary>
    /// Shrink applied to the search range after every sweep. Below 1 this
    /// zooms in on the current peak, giving coarse-to-fine behaviour without
    /// separate stages: early sweeps explore the box, later sweeps refine.
    /// </summary>
    private const float ZOOM_FACTOR = 0.6f;

    /// <summary>
    /// Smallest search range worth using, in mm. The range is clamped here so
    /// the step never collapses into denormal territory, which would waste the
    /// remaining steps and make the sweep positions numerically meaningless.
    /// </summary>
    private const float MIN_RANGE_MM = 1f;

    /// <summary>Smallest search range worth using, in degrees.</summary>
    private const float MIN_ANGLE_RANGE_DEG = 0.1f;

    /// <summary>
    /// Fraction of the available score headroom the search must capture to
    /// accept the estimate. Normalising by the headroom makes the test
    /// scale-invariant, so a scan dominated by obstacles (which caps the
    /// achievable score well below 1) is not rejected for that reason alone.
    /// </summary>
    private const float MIN_SCORE_GAIN_FRACTION = 0.02f;

    /// <summary>True when the last estimate failed and the approximate position was kept.</summary>
    public static bool LastEstimateWasRejected { get; private set; }

    /// <summary>Reason the last estimate was rejected (empty when accepted).</summary>
    public static string LastRejectionReason { get; private set; } = string.Empty;

    /// <summary>Wall-clock duration of the last estimate, in milliseconds.</summary>
    public static double LastEstimateMs { get; private set; }

    /// <summary>Sweeps the last estimate actually ran before converging.</summary>
    public static int LastSweepCount { get; private set; }

    /// <summary>
    /// Bakes the distance-to-nearest-wall field. Called once, lazily, before the
    /// first estimate. Each node stores the clamped distance so the far field
    /// saturates instead of growing without bound - beyond
    /// FIELD_MAX_DISTANCE_MM the exact value never affects the score.
    /// </summary>
    private static void BuildDistanceField()
    {
        short[] field = new short[FIELD_SIZE * FIELD_SIZE];

        for (int iy = 0; iy < FIELD_SIZE; iy++)
        {
            float y = -FIELD_HALF_MM + iy * FIELD_CELL_MM;

            for (int ix = 0; ix < FIELD_SIZE; ix++)
            {
                float x = -FIELD_HALF_MM + ix * FIELD_CELL_MM;

                float distance = DistanceToNearestWall(new Vector2(x, y), out _, out _);
                distance = Mathf.Min(distance, FIELD_MAX_DISTANCE_MM);

                field[iy * FIELD_SIZE + ix] = (short)Mathf.RoundToInt(distance / FIELD_UNIT_MM);
            }
        }

        distance_field = field;
    }

    /// <summary>
    /// Distance to the nearest wall at a point, in mm, by bilinear interpolation
    /// of the baked field. This is the hot path: it replaces a loop over every
    /// wall segment with four loads and a handful of multiplies.
    /// </summary>
    private static float SampleDistanceField(float x, float y)
    {
        float fx = (x + FIELD_HALF_MM) / FIELD_CELL_MM;
        float fy = (y + FIELD_HALF_MM) / FIELD_CELL_MM;

        // Outside the baked grid: clamp to the border, whose value is the
        // saturated distance, so the score is uniformly zero out there.
        if (fx <= 0f || fy <= 0f || fx >= FIELD_SIZE - 1 || fy >= FIELD_SIZE - 1)
        {
            return FIELD_MAX_DISTANCE_MM;
        }

        int ix = (int)fx;
        int iy = (int)fy;

        float tx = fx - ix;
        float ty = fy - iy;

        int row = iy * FIELD_SIZE + ix;

        // Interpolate in the stored units, then scale once at the end.
        float d00 = distance_field[row];
        float d10 = distance_field[row + 1];
        float d01 = distance_field[row + FIELD_SIZE];
        float d11 = distance_field[row + FIELD_SIZE + 1];

        // Bilinear blend; the compiler vectorises the fixed mixes.
        float top = d00 + (d10 - d00) * tx;
        float bottom = d01 + (d11 - d01) * tx;

        return (top + (bottom - top) * ty) * FIELD_UNIT_MM;
    }

    /// <summary>
    /// Distance from a point to a field wall segment, clamped to the segment ends.
    /// Also returns the wall's direction and normal. This is the exact routine,
    /// used only to bake the distance field.
    /// </summary>
    private static float DistanceToNearestWall(Vector2 point, out Vector2 wall_direction, out Vector2 wall_normal)
    {
        float best_distance = float.MaxValue;
        wall_direction = Vector2.right;
        wall_normal = Vector2.up;

        for (int i = 0; i < FIELD_OUTLINE.Length; i++)
        {
            Vector2 a = FIELD_OUTLINE[i];
            Vector2 b = FIELD_OUTLINE[(i + 1) % FIELD_OUTLINE.Length];

            Vector2 edge = b - a;
            float edge_length = edge.magnitude;

            if (edge_length < Mathf.Epsilon)
            {
                continue;
            }

            Vector2 direction = edge / edge_length;
            Vector2 normal = new Vector2(-direction.y, direction.x);

            // Project onto the segment, clamped to its ends, so the distance is
            // measured to the wall itself and not to its infinite extension.
            float t = Mathf.Clamp(Vector2.Dot(point - a, direction) / edge_length, 0f, 1f);
            Vector2 closest = a + direction * (t * edge_length);
            float distance = Vector2.Distance(point, closest);

            if (distance < best_distance)
            {
                best_distance = distance;
                wall_direction = direction;
                wall_normal = normal;
            }
        }

        return best_distance;
    }

    /// <summary>
    /// Refines the robot's position with correlation scan matching.
    ///
    /// The pose is recovered one axis at a time (heading, then X, then Y), so
    /// the search cost grows linearly with the grid resolution instead of
    /// cubically. Every candidate is still a full robot pose, projected from the
    /// robot's centre, so no pivot bookkeeping or frame conversion is needed.
    /// </summary>
    /// <param name="approximate_position">Best guess of the robot's position, e.g. from odometry.</param>
    /// <param name="measurements">Latest lidar scan.</param>
    /// <param name="lidar_offset">Lidar position relative to the robot's centre.</param>
    /// <returns>The estimated real position.</returns>
    public static Pos EstimatePosition(
        Pos approximate_position,
        List<Lidar.Measurement> measurements,
        Pos lidar_offset)
    {
        LastEstimateWasRejected = false;
        LastRejectionReason = string.Empty;

        if (measurements.Count < 2)
        {
            Reject("scan produced fewer than 2 points");
            return approximate_position;
        }

        if (distance_field == null)
        {
            BuildDistanceField();
        }

        PrepareScan(measurements);

        SearchWorkspace workspace = new SearchWorkspace(measurements.Count);

        System.Diagnostics.Stopwatch clock = System.Diagnostics.Stopwatch.StartNew();

        float base_score = ScorePose(measurements, approximate_position, lidar_offset, workspace);

        // --- Zooming coordinate-descent search ------------------------------
        PoseEstimate best = Search(measurements, approximate_position, lidar_offset, workspace);

        clock.Stop();
        LastEstimateMs = clock.Elapsed.TotalMilliseconds;
        LastSweepCount = workspace.sweeps_run;

        float best_x = best.x;
        float best_y = best.y;
        float best_a = best.a;
        float best_score = best.score;

        // --- Acceptance ------------------------------------------------------
        // Reject only when the search genuinely failed to improve on the input.
        // The gain is measured against the headroom left above the base score,
        // so it does not depend on how high the achievable score is for this
        // particular scan.
        float headroom = 1f - base_score;
        float relative_gain = headroom > Mathf.Epsilon
            ? (best_score - base_score) / headroom
            : 0f;

        if (relative_gain < MIN_SCORE_GAIN_FRACTION)
        {
            Reject($"search found no improvement (base score {base_score:F3}, " +
                   $"best {best_score:F3}, gain {relative_gain:P1} of {headroom:F3} headroom " +
                   $"below {MIN_SCORE_GAIN_FRACTION:P1})");
            return approximate_position;
        }

        Pos estimated_position = new Pos { pos_x = best_x, pos_y = best_y, pos_a = best_a };

        return estimated_position;
    }

    /// <summary>
    /// Bakes the per-ray invariants for one scan: the range in mm and the sine
    /// and cosine of each ray's bearing. These depend only on the scan, not on
    /// the candidate pose, but <see cref="ProjectScan"/> runs many times per
    /// estimate, so recomputing them there was pure waste.
    /// </summary>
    private static void PrepareScan(List<Lidar.Measurement> measurements)
    {
        int count = measurements.Count;

        if (scan_cos == null || scan_cos.Length != count)
        {
            scan_cos = new float[count];
            scan_sin = new float[count];
            scan_range_mm = new float[count];
        }

        for (int i = 0; i < count; i++)
        {
            float angle_rad = measurements[i].angle * Mathf.Deg2Rad;

            scan_cos[i] = Mathf.Cos(angle_rad);
            scan_sin[i] = Mathf.Sin(angle_rad);
            scan_range_mm[i] = measurements[i].distance * 1000f;
        }
    }

    /// <summary>
    /// Projects the scan from a candidate robot pose into field coordinates.
    /// The lidar sits at an offset from the robot's centre, so that offset is
    /// rotated by the candidate heading before the rays are cast.
    ///
    /// Each ray's range in mm and the sine/cosine of its own bearing are baked
    /// once per scan by <see cref="PrepareScan"/>, so this loop is only the
    /// per-pose rotation and translation.
    /// </summary>
    private static Vector2[] ProjectScan(
        List<Lidar.Measurement> measurements,
        Pos pose,
        Pos lidar_offset,
        Vector2[] buffer)
    {
        float a_rad = pose.pos_a * Mathf.Deg2Rad;
        float cos_a = Mathf.Cos(a_rad);
        float sin_a = Mathf.Sin(a_rad);

        // Lidar origin in field coordinates.
        float origin_x = pose.pos_x + lidar_offset.pos_x * cos_a - lidar_offset.pos_y * sin_a;
        float origin_y = pose.pos_y + lidar_offset.pos_x * sin_a + lidar_offset.pos_y * cos_a;

        // Rotating each ray by the pose is a single complex multiply against the
        // precomputed bearing, so no per-ray trig remains here.
        for (int i = 0; i < measurements.Count; i++)
        {
            float local_x = scan_cos[i];
            float local_y = scan_sin[i];
            float range_mm = scan_range_mm[i];

            buffer[i] = new Vector2(
                origin_x + range_mm * (local_x * cos_a - local_y * sin_a),
                origin_y + range_mm * (local_x * sin_a + local_y * cos_a));
        }

        return buffer;
    }

    /// <summary>
    /// Scores the scan projected from a candidate robot pose.
    /// </summary>
    private static float ScorePose(
        List<Lidar.Measurement> measurements,
        Pos pose,
        Pos lidar_offset,
        SearchWorkspace workspace)
    {
        ProjectScan(measurements, pose, lidar_offset, workspace.base_points);
        return ScoreProjected(workspace.base_points);
    }

    /// <summary>
    /// Mean Gaussian distance-to-wall score over every projected point. The
    /// falloff is smooth, so the score has a gradient pointing toward the
    /// correct pose instead of a hard inlier cliff.
    /// </summary>
    private static float ScoreProjected(Vector2[] points)
    {
        float score = 0f;
        int count = points.Length;

        for (int i = 0; i < count; i++)
        {
            score += PointScore(SampleDistanceField(points[i].x, points[i].y));
        }

        return count > 0 ? score / count : 0f;
    }

    /// <summary>Outcome of one search schedule.</summary>
    private struct PoseEstimate
    {
        public float x;
        public float y;
        public float a;
        public float score;
    }

    /// <summary>Scratch buffer for one schedule, reused by every sweep.</summary>
    private sealed class SearchWorkspace
    {
        // Projected scan for the best pose found so far.
        public readonly Vector2[] base_points;

        /// <summary>Sweeps the last search actually ran, for diagnostics.</summary>
        public int sweeps_run;

        public SearchWorkspace(int rays)
        {
            base_points = new Vector2[rays];
        }
    }

    /// <summary>
    /// Runs a zooming coordinate-descent search: repeatedly sweeps the heading,
    /// then X, then Y, sampling POINTS_PER_SWEEP either side of the current best
    /// on each axis, and shrinking the search range by ZOOM_FACTOR after every
    /// sweep.
    ///
    /// There are no separate coarse and fine stages: the early sweeps explore
    /// the whole RANGE_MM box with a wide step, and the zoom turns the same
    /// sweep into ever finer refinement of the peak it has found. Each sweep
    /// keeps the other two coordinates fixed, so a sweep costs only
    /// 3*(2*POINTS_PER_SWEEP+1) pose evaluations regardless of how coarse or
    /// fine it is.
    ///
    /// The range is clamped to a floor so the step never collapses to zero.
    /// TOTAL_STEPS sweeps always run; there is no early exit, so the cost is
    /// constant and the zoom is guaranteed to reach its narrowest step.
    /// </summary>
    private static PoseEstimate Search(
        List<Lidar.Measurement> measurements,
        Pos centre,
        Pos lidar_offset,
        SearchWorkspace workspace)
    {
        PoseEstimate best = new PoseEstimate
        {
            x = centre.pos_x,
            y = centre.pos_y,
            a = centre.pos_a,
            score = ScorePose(measurements, centre, lidar_offset, workspace),
        };

        float range_mm = RANGE_MM;
        float range_deg = ANGLE_RANGE_DEG;

        for (int step_index = 0; step_index < TOTAL_STEPS; step_index++)
        {
            float xy_step = range_mm / POINTS_PER_SWEEP;
            float angle_step = range_deg / POINTS_PER_SWEEP;

            SweepAngle(measurements, lidar_offset, workspace, POINTS_PER_SWEEP, angle_step, ref best);
            SweepX(measurements, lidar_offset, workspace, POINTS_PER_SWEEP, xy_step, ref best);
            SweepY(measurements, lidar_offset, workspace, POINTS_PER_SWEEP, xy_step, ref best);

            workspace.sweeps_run = step_index + 1;

            // Zoom in on the peak, never below the smallest useful range.
            range_mm = Mathf.Max(MIN_RANGE_MM, range_mm * ZOOM_FACTOR);
            range_deg = Mathf.Max(MIN_ANGLE_RANGE_DEG, range_deg * ZOOM_FACTOR);
        }

        return best;
    }

    /// <summary>
    /// Sweeps X with Y and the heading fixed.
    ///
    /// Moving the robot along X translates the whole projected scan along X, so
    /// the scan is projected once and each candidate is a cheap shift of it.
    /// </summary>
    private static void SweepX(
        List<Lidar.Measurement> measurements,
        Pos lidar_offset,
        SearchWorkspace workspace,
        int steps,
        float step,
        ref PoseEstimate best)
    {
        // Projected with the robot at x = 0; adding the candidate x to every
        // point (and to the lidar origin) is the same as moving the robot.
        ProjectScan(measurements, new Pos { pos_x = 0f, pos_y = best.y, pos_a = best.a },
                    lidar_offset, workspace.base_points);

        float best_x = best.x;
        float best_score = best.score;

        for (int xi = -steps; xi <= steps; xi++)
        {
            float x = best.x + xi * step;
            float score = 0f;

            for (int i = 0; i < workspace.base_points.Length; i++)
            {
                Vector2 point = workspace.base_points[i];
                point.x += x;

                score += PointScore(SampleDistanceField(point.x, point.y));
            }

            score /= workspace.base_points.Length;

            if (score > best_score)
            {
                best_score = score;
                best_x = x;
            }
        }

        best.x = best_x;
        best.score = best_score;
    }

    /// <summary>
    /// Sweeps Y with X and the heading fixed, again by translating the scan.
    /// Every ray is evaluated per candidate; the score has no cutoff, so even a
    /// far ray still contributes a small amount and cannot be skipped.
    /// </summary>
    private static void SweepY(
        List<Lidar.Measurement> measurements,
        Pos lidar_offset,
        SearchWorkspace workspace,
        int steps,
        float step,
        ref PoseEstimate best)
    {
        ProjectScan(measurements, new Pos { pos_x = best.x, pos_y = 0f, pos_a = best.a },
                    lidar_offset, workspace.base_points);

        float best_y = best.y;
        float best_score = best.score;

        for (int yi = -steps; yi <= steps; yi++)
        {
            float y = best.y + yi * step;
            float score = 0f;

            for (int i = 0; i < workspace.base_points.Length; i++)
            {
                Vector2 point = workspace.base_points[i];
                point.y += y;
                score += PointScore(SampleDistanceField(point.x, point.y));
            }

            score /= workspace.base_points.Length;

            if (score > best_score)
            {
                best_score = score;
                best_y = y;
            }
        }

        best.y = best_y;
        best.score = best_score;
    }

    /// <summary>
    /// Sweeps the heading with X and Y fixed.
    ///
    /// Rotating the robot swings every ray, and the lidar offset makes each
    /// projected point travel on an arc about the robot's centre. The score has
    /// no plateau, so every point changes score under rotation and all rays are
    /// evaluated per candidate.
    /// </summary>
    private static void SweepAngle(
        List<Lidar.Measurement> measurements,
        Pos lidar_offset,
        SearchWorkspace workspace,
        int steps,
        float step,
        ref PoseEstimate best)
    {
        Pos best_pose = new Pos { pos_x = best.x, pos_y = best.y, pos_a = best.a };
        ProjectScan(measurements, best_pose, lidar_offset, workspace.base_points);

        float centre_x = best.x;
        float centre_y = best.y;

        float best_a = best.a;
        float best_score = best.score;

        for (int ai = -steps; ai <= steps; ai++)
        {
            float angle = best.a + ai * step;
            float delta = (angle - best.a) * Mathf.Deg2Rad;
            float cos_d = Mathf.Cos(delta);
            float sin_d = Mathf.Sin(delta);

            float score = 0f;

            for (int i = 0; i < workspace.base_points.Length; i++)
            {
                // Rotate the point about the robot centre by delta.
                Vector2 point = workspace.base_points[i];
                float offset_x = point.x - centre_x;
                float offset_y = point.y - centre_y;

                Vector2 rotated = new Vector2(
                    centre_x + offset_x * cos_d - offset_y * sin_d,
                    centre_y + offset_x * sin_d + offset_y * cos_d);

                score += PointScore(SampleDistanceField(rotated.x, rotated.y));
            }

            score /= workspace.base_points.Length;

            if (score > best_score)
            {
                best_score = score;
                best_a = angle;
            }
        }

        best.a = best_a;
        best.score = best_score;
    }

    /// <summary>
    /// Score contribution of a single point at a given distance from the nearest
    /// wall: a plain Gaussian, maximal on the wall and never exactly zero.
    /// </summary>
    private static float PointScore(float distance)
    {
        float sigma = SCORE_SIGMA_MM;
        return Mathf.Exp(-(distance * distance) / (2f * sigma * sigma));
    }

    /// <summary>
    /// Records that the estimate failed. The returned position is then the
    /// unchanged approximate position, which is indistinguishable from a perfect
    /// estimate, so this is deliberately loud.
    /// </summary>
    private static void Reject(string reason)
    {
        LastEstimateWasRejected = true;
        LastRejectionReason = reason;
        Debug.LogError($"PosEstimator REJECTED the scan estimate: {reason}. " +
                       $"Returning the approximate position unchanged, which will look " +
                       $"like a perfect estimate at zero odometry noise.");
    }
}
