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

    // --- Search schedule ---------------------------------------------------
    /// <summary>Half-width of the initial search box around the approximate pose, in mm.</summary>
    private const float RANGE_MM = 300f;

    /// <summary>Half-width of the initial search box around the approximate heading, in degrees.</summary>
    private const float ANGLE_RANGE_DEG = 30f;

    /// <summary>Total number of coordinate-descent sweeps performed per estimate.</summary>
    private const int TOTAL_STEPS = 10;

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
    /// Stop early once a sweep improves the score by less than this, as a
    /// guard against burning the remaining steps on a converged result.
    /// </summary>
    private const float SWEEP_CONVERGENCE_EPSILON = 1e-3f;

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

    /// <summary>
    /// Distance from a point to a field wall segment, clamped to the segment ends.
    /// Also returns the wall's direction and normal.
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

        float base_score = ScorePose(measurements, approximate_position, lidar_offset);

        // --- Zooming coordinate-descent search ------------------------------
        PoseEstimate best = Search(measurements, approximate_position, lidar_offset);

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
    /// Projects the scan from a candidate robot pose into field coordinates.
    /// The lidar sits at an offset from the robot's centre, so that offset is
    /// rotated by the candidate heading before the rays are cast.
    /// </summary>
    private static Vector2[] ProjectScan(
        List<Lidar.Measurement> measurements,
        Pos pose,
        Pos lidar_offset,
        Vector2[] buffer)
    {
        float cos_a = Mathf.Cos(pose.pos_a * Mathf.Deg2Rad);
        float sin_a = Mathf.Sin(pose.pos_a * Mathf.Deg2Rad);

        // Lidar origin in field coordinates.
        float origin_x = pose.pos_x + lidar_offset.pos_x * cos_a - lidar_offset.pos_y * sin_a;
        float origin_y = pose.pos_y + lidar_offset.pos_x * sin_a + lidar_offset.pos_y * cos_a;

        for (int i = 0; i < measurements.Count; i++)
        {
            float range_mm = measurements[i].distance * 1000f;
            float angle_rad = (pose.pos_a + measurements[i].angle) * Mathf.Deg2Rad;

            buffer[i] = new Vector2(
                origin_x + range_mm * Mathf.Cos(angle_rad),
                origin_y + range_mm * Mathf.Sin(angle_rad));
        }

        return buffer;
    }

    /// <summary>Scores the scan projected from a candidate robot pose.</summary>
    private static float ScorePose(
        List<Lidar.Measurement> measurements,
        Pos pose,
        Pos lidar_offset)
    {
        Vector2[] buffer = new Vector2[measurements.Count];
        return ScoreProjected(ProjectScan(measurements, pose, lidar_offset, buffer));
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
    /// The range is clamped to a floor so the step never collapses to zero and
    /// the search keeps making progress until TOTAL_STEPS is exhausted.
    /// </summary>
    private static PoseEstimate Search(
        List<Lidar.Measurement> measurements,
        Pos centre,
        Pos lidar_offset)
    {
        SearchWorkspace workspace = new SearchWorkspace(measurements.Count);

        PoseEstimate best = new PoseEstimate
        {
            x = centre.pos_x,
            y = centre.pos_y,
            a = centre.pos_a,
            score = ScorePose(measurements, centre, lidar_offset),
        };

        float range_mm = RANGE_MM;
        float range_deg = ANGLE_RANGE_DEG;

        for (int step_index = 0; step_index < TOTAL_STEPS; step_index++)
        {
            float sweep_start_score = best.score;

            float xy_step = range_mm / POINTS_PER_SWEEP;
            float angle_step = range_deg / POINTS_PER_SWEEP;

            SweepAngle(measurements, lidar_offset, workspace, POINTS_PER_SWEEP, angle_step, ref best);
            SweepX(measurements, lidar_offset, workspace, POINTS_PER_SWEEP, xy_step, ref best);
            SweepY(measurements, lidar_offset, workspace, POINTS_PER_SWEEP, xy_step, ref best);

            float gain = best.score - sweep_start_score;

            // Zoom in on the peak, never below the smallest useful range.
            range_mm = Mathf.Max(MIN_RANGE_MM, range_mm * ZOOM_FACTOR);
            range_deg = Mathf.Max(MIN_ANGLE_RANGE_DEG, range_deg * ZOOM_FACTOR);

            if (gain < SWEEP_CONVERGENCE_EPSILON)
            {
                break;
            }
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

                score += PointScore(DistanceToNearestWall(point, out _, out _));
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
                score += PointScore(DistanceToNearestWall(point, out _, out _));
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

                score += PointScore(DistanceToNearestWall(rotated, out _, out _));
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
    /// Scores projected points by the mean Gaussian distance-to-wall score. The
    /// falloff is smooth, so the score has a gradient pointing toward the
    /// correct pose instead of a hard inlier cliff.
    /// </summary>
    private static float ScoreProjected(Vector2[] points)
    {
        float score = 0f;

        for (int i = 0; i < points.Length; i++)
        {
            score += PointScore(DistanceToNearestWall(points[i], out _, out _));
        }

        return points.Length > 0 ? score / points.Length : 0f;
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
