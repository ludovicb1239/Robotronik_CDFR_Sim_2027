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
/// Scoring: each ray scores a robust kernel of the distance d from its hit
/// point to the nearest field wall, and the pose score is the mean over rays.
/// The kernel is 1 on a wall and decays smoothly, so a point near a wall
/// carries strong positional information while a point far from every wall -
/// which is what an obstacle return looks like - contributes a small, bounded
/// pull rather than either a hard zero or an outsized vote. Crucially it never
/// becomes exactly flat, so the search always has a gradient to follow even
/// when the whole scan starts well away from the walls.
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
/// 1. the field has only eight boundary segments, so the exact distance from a
///    point to the nearest wall is eight clamped projections - cheap enough
///    that no baked distance field or spatial index is needed, and exact rather
///    than quantised and interpolated;
/// 2. each ray's range and bearing are baked once per scan, so projecting the
///    scan carries no per-ray trigonometry and each candidate pose is a single
///    rotation and translation of the already-computed bearing;
/// 3. the score is a bounded rational kernel, which the compiler turns into a
///    multiply-add, a divide and no transcendental call at all.
///
/// Note on obstacle rays: they are deliberately still scored, not filtered out.
/// Filtering them was tried and regressed, because an obstacle return is a
/// repeatable feature rather than noise and dropping it starves a 3-DOF fit of
/// constraints. The robust kernel achieves the useful part of that idea - far
/// points stop steering the search - without the cost of deleting evidence.
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
        new Vector2(  -550f, -900f),
        new Vector2(  -550f,  900f),
        new Vector2( -1000f,  900f),
        new Vector2( -1000f, 1500f),
    };

    /// <summary>Falloff of the score with distance from a wall, in mm.</summary>
    private const float SCORE_SIGMA_MM = 10f;

    /// <summary>
    /// Scale at which the score stops falling appreciably, in mm.
    ///
    /// A plain Gaussian keeps decreasing forever, so a projected point that
    /// sits on an obstacle far from every wall scores almost exactly zero and
    /// drags the mean down from the correct pose toward whatever pose happens
    /// to push obstacle returns onto walls. The score therefore has to become
    /// insensitive to distance out there, so that obstacle returns stop
    /// steering the search.
    ///
    /// The insensitivity is achieved by softening the Gaussian into a robust
    /// kernel rather than by clamping the distance. Clamping was tried and was
    /// a regression: it makes the score exactly flat beyond the clamp, so a ray
    /// already past it contributes no gradient at all, and the search can only
    /// escape such a plateau if some other ray happens to still be inside the
    /// clamp radius. With a prior of +/-75 mm and a 60 mm clamp, roughly half of
    /// all scans started with every wall ray already flat, and those scans
    /// simply kept the prior - which is the loss of precision this kernel fixes.
    ///
    /// The kernel below is a Geman-McClure form. It falls off like a Gaussian
    /// near a wall, where the positional information lives, but its tails decay
    /// only quadratically, so a far point contributes a small, monotonically
    /// decreasing pull toward the wall instead of a constant. There is no region
    /// of exactly zero gradient, so the search always has a direction to move.
    /// The influence of a far point is bounded well below that of a wall point,
    /// which is what keeps obstacle returns from dominating the fit.
    /// </summary>
    private const float SCORE_ROLLOFF_MM = 2.5f * SCORE_SIGMA_MM;

    // --- Field geometry ----------------------------------------------------
    /// <summary>
    /// The boundary segments, flattened into parallel arrays so the distance
    /// walk can move along them without an indirection back through the
    /// polygon's vertex list or a modulo to find the wrap-around edge.
    ///
    /// This replaces the baked distance-to-nearest-wall grid. The grid bought
    /// speed by turning a walk over the segments into a bilinear lookup, but it
    /// paid for that with a 2001x2001 short array (about 7.6 MB), a one-off
    /// bake, a 0.1 mm quantisation, and an interpolation error of a fraction of
    /// a 2 mm cell. None of that is necessary: there are only eight segments, so
    /// exact geometry is a handful of arithmetic per segment and is both easier
    /// to reason about and exactly correct, with no quantisation and no
    /// interpolation error anywhere.
    /// </summary>
    private static readonly Vector2[] segment_start;
    private static readonly Vector2[] segment_edge;
    private static readonly float[] segment_length_squared;

    /// <summary>
    /// Fills the flattened segment tables from <see cref="FIELD_OUTLINE"/>. The
    /// polygon's last vertex is wired back to the first, so the closing edge is
    /// present in the table like any other and the distance walk needs no
    /// wrap-around special case.
    /// </summary>
    static PosEstimator()
    {
        int count = FIELD_OUTLINE.Length;

        segment_start = new Vector2[count];
        segment_edge = new Vector2[count];
        segment_length_squared = new float[count];

        for (int i = 0; i < count; i++)
        {
            Vector2 a = FIELD_OUTLINE[i];
            Vector2 b = FIELD_OUTLINE[(i + 1) % count];
            Vector2 edge = b - a;

            segment_start[i] = a;
            segment_edge[i] = edge;
            segment_length_squared[i] = edge.sqrMagnitude;
        }
    }

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
    /// Distance from a point to the nearest field wall segment, clamped to the
    /// segment ends, by walking all eight boundary edges.
    ///
    /// This is exact, unlike the interpolated lookup it replaces. It is called
    /// once per ray per candidate pose, so it is the hot path; eight segment
    /// projections is small enough that no acceleration structure is warranted,
    /// and keeping it exact removes the quantisation and interpolation error the
    /// baked field carried while also dropping the multimegabyte array.
    /// </summary>
    private static float DistanceToNearestWall(Vector2 point)
    {
        float best_distance_squared = float.MaxValue;

        for (int i = 0; i < segment_start.Length; i++)
        {
            float ex = segment_edge[i].x;
            float ey = segment_edge[i].y;

            float ax = point.x - segment_start[i].x;
            float ay = point.y - segment_start[i].y;

            // Project onto the segment and clamp to its ends, so the distance is
            // measured to the wall itself and not to its infinite extension.
            float length_squared = segment_length_squared[i];

            if (length_squared <= 0f)
            {
                continue;
            }

            float t = (ax * ex + ay * ey) / length_squared;
            t = t < 0f ? 0f : (t > 1f ? 1f : t);

            float dx = ax - t * ex;
            float dy = ay - t * ey;
            float distance_squared = dx * dx + dy * dy;

            if (distance_squared < best_distance_squared)
            {
                best_distance_squared = distance_squared;
            }
        }

        return Mathf.Sqrt(best_distance_squared);
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
    /// Mean score over every projected point, where a point's score is the
    /// robust kernel of the distance from that point to the nearest wall.
    ///
    /// The distance is measured to the nearest of the eight boundary segments,
    /// found by walking them, which is exact where the baked distance field used
    /// to interpolate. Note that this is a point-to-wall distance, not a
    /// comparison of measured and predicted range: the two agree only when the
    /// ray truly terminated on a wall. For an obstacle return they differ, which
    /// is deliberate - the kernel makes such a point contribute a small bounded
    /// pull rather than pretending the wall was closer than it is.
    /// </summary>
    private static float ScoreProjected(Vector2[] points)
    {
        float score = 0f;
        int count = points.Length;

        for (int i = 0; i < count; i++)
        {
            score += PointScore(DistanceToNearestWall(points[i]));
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

                score += PointScore(DistanceToNearestWall(point));
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
    /// Every ray is evaluated per candidate, because the mean is taken over the
    /// full ray count and every ray contributes some gradient.
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
                score += PointScore(DistanceToNearestWall(point));
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
    /// projected point travel on an arc about the robot's centre. Points near a
    /// wall change score fastest under rotation and so drive this sweep most,
    /// but because the kernel never becomes flat every ray contributes some
    /// gradient, which is what lets a scan that starts far from the walls still
    /// rotate toward them.
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

                score += PointScore(DistanceToNearestWall(rotated));
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
    /// wall, as a Geman-McClure robust kernel:
    ///
    ///     s(d) = rolloff^2 / (rolloff^2 + 2 * d^2)
    ///
    /// Near a wall this behaves like the Gaussian it replaces - it is 1 on the
    /// wall, halves at about 0.64 * rolloff, and its sharpest slope sits a little
    /// inside rolloff - so the positional information that locates the robot is
    /// unchanged. Far from a wall it decays like 1/d^2 instead of collapsing to
    /// exp(-large), which is the whole point: an obstacle return is far from
    /// every wall, so it contributes a small, slowly-varying pull rather than a
    /// hard zero, and it can never outvote the wall points that sit at distance
    /// near zero and score near 1.
    ///
    /// Two properties matter for the search. First, the function is strictly
    /// decreasing in d everywhere, with no flat region, so every ray retains a
    /// direction to improve and a pose whose wall rays all start far away can
    /// still be pulled in. That is what the previous hard clamp destroyed.
    /// Second, the kernel is bounded, so no single bad cluster of rays can drive
    /// the score, which is why a robust kernel is used here rather than a plain
    /// Gaussian plus an outlier cutoff.
    /// </summary>
    private static float PointScore(float distance)
    {
        float rolloff_squared = SCORE_ROLLOFF_MM * SCORE_ROLLOFF_MM;
        return rolloff_squared / (rolloff_squared + 2f * distance * distance);
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
