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
/// Algorithm: correlation scan matching. Instead of randomly sampling features,
/// search a small box of (dx, dy, da) around the approximate pose and maximise
/// a smooth likelihood-field score. The walls are known exactly, the prior is
/// tight, and obstacles simply contribute a low score rather than breaking a
/// hypothesis. The search is coarse-to-fine so it cannot miss the true pose
/// inside the box, then a parabola fit gives sub-cell accuracy.
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

    /// <summary>Distance below which a point scores full marks, in mm.</summary>
    private const float SCORE_PLATEAU_MM = 10f;

    /// <summary>Falloff of the score with distance from a wall, in mm.</summary>
    private const float SCORE_SIGMA_MM = 40f;

    /// <summary>Point further than this from any wall score nothing, in mm.</summary>
    private const float SCORE_CUTOFF_MM = 200f;

    // --- Coarse pass ---
    private const float COARSE_XY_RANGE_MM = 200f;
    private const float COARSE_XY_STEP_MM = 25f;
    private const float COARSE_ANGLE_RANGE_DEG = 10f;
    private const float COARSE_ANGLE_STEP_DEG = 1f;

    // --- Fine pass, centred on the coarse winner ---
    private const float FINE_XY_RANGE_MM = 60f;
    private const float FINE_XY_STEP_MM = 5f;
    private const float FINE_ANGLE_RANGE_DEG = 1.5f;
    private const float FINE_ANGLE_STEP_DEG = 0.2f;

    /// <summary>Score of the approximate position must be beaten by this much to accept.</summary>
    private const float MIN_SCORE_GAIN = 0.02f;

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
    /// Every candidate is a full robot pose, and its point cloud is re-projected
    /// from that pose, so the rotation happens about the robot's centre and the
    /// pose that maximises the score is the answer - no pivot bookkeeping and no
    /// frame conversion on the way out.
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

        float best_x = approximate_position.pos_x;
        float best_y = approximate_position.pos_y;
        float best_a = approximate_position.pos_a;
        float best_score = base_score;

        // --- Coarse pass: cover the whole expected odometry error -----------
        Search(measurements, approximate_position, lidar_offset,
               COARSE_XY_RANGE_MM, COARSE_XY_STEP_MM,
               COARSE_ANGLE_RANGE_DEG, COARSE_ANGLE_STEP_DEG,
               ref best_x, ref best_y, ref best_a, ref best_score);

        // --- Fine pass: refine around the coarse winner ---------------------
        float coarse_score = best_score;
        Pos coarse_pose = new Pos { pos_x = best_x, pos_y = best_y, pos_a = best_a };

        Search(measurements, coarse_pose, lidar_offset,
               FINE_XY_RANGE_MM, FINE_XY_STEP_MM,
               FINE_ANGLE_RANGE_DEG, FINE_ANGLE_STEP_DEG,
               ref best_x, ref best_y, ref best_a, ref best_score);

        // --- Sub-cell refinement --------------------------------------------
        // Fit a parabola through the winner and its neighbours on each axis so
        // the result is not quantised to the grid step.
        best_x += ParabolaOffset(
            ScorePose(measurements, OffsetPose(best_x, best_y, best_a, -FINE_XY_STEP_MM, 0f, 0f), lidar_offset),
            best_score,
            ScorePose(measurements, OffsetPose(best_x, best_y, best_a, FINE_XY_STEP_MM, 0f, 0f), lidar_offset),
            FINE_XY_STEP_MM);

        best_y += ParabolaOffset(
            ScorePose(measurements, OffsetPose(best_x, best_y, best_a, 0f, -FINE_XY_STEP_MM, 0f), lidar_offset),
            best_score,
            ScorePose(measurements, OffsetPose(best_x, best_y, best_a, 0f, FINE_XY_STEP_MM, 0f), lidar_offset),
            FINE_XY_STEP_MM);

        best_a += ParabolaOffset(
            ScorePose(measurements, OffsetPose(best_x, best_y, best_a, 0f, 0f, -FINE_ANGLE_STEP_DEG), lidar_offset),
            best_score,
            ScorePose(measurements, OffsetPose(best_x, best_y, best_a, 0f, 0f, FINE_ANGLE_STEP_DEG), lidar_offset),
            FINE_ANGLE_STEP_DEG);

        float improved = best_score - base_score;

        // --- Acceptance ------------------------------------------------------
        // Reject only when the search genuinely failed to improve on the input.
        // A low absolute score is fine: it just means many rays hit obstacles.
        if (improved < MIN_SCORE_GAIN)
        {
            Reject($"search found no improvement (base score {base_score:F3}, " +
                   $"best {best_score:F3}, gain {improved:F3} below {MIN_SCORE_GAIN:F3}; " +
                   $"coarse pass reached {coarse_score:F3})");
            return approximate_position;
        }

        Pos estimated_position = new Pos { pos_x = best_x, pos_y = best_y, pos_a = best_a };

        Debug.Log($"PosEstimator: {measurements.Count} rays, score {base_score:F3} -> {best_score:F3}, " +
                  $"correction (dx {best_x - approximate_position.pos_x:F0}, " +
                  $"dy {best_y - approximate_position.pos_y:F0}, " +
                  $"da {best_a - approximate_position.pos_a:F2}), " +
                  $"estimated ({estimated_position.pos_x:F0}, {estimated_position.pos_y:F0}, {estimated_position.pos_a:F2})");

        return estimated_position;
    }

    /// <summary>Returns a copy of a pose with an offset applied to each component.</summary>
    private static Pos OffsetPose(float x, float y, float a, float dx, float dy, float da)
    {
        return new Pos { pos_x = x + dx, pos_y = y + dy, pos_a = a + da };
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

    /// <summary>
    /// Scores projected points. Each point contributes 1 when on a wall, then
    /// falls off smoothly with distance, so the score has a gradient pointing
    /// toward the correct pose instead of a hard inlier cliff.
    /// </summary>
    private static float ScoreProjected(Vector2[] points)
    {
        float score = 0f;
        float two_sigma_sq = 2f * SCORE_SIGMA_MM * SCORE_SIGMA_MM;

        for (int i = 0; i < points.Length; i++)
        {
            float distance = DistanceToNearestWall(points[i], out _, out _);

            if (distance <= SCORE_PLATEAU_MM)
            {
                score += 1f;
            }
            else if (distance < SCORE_CUTOFF_MM)
            {
                float excess = distance - SCORE_PLATEAU_MM;
                score += Mathf.Exp(-(excess * excess) / two_sigma_sq);
            }
        }

        return points.Length > 0 ? score / points.Length : 0f;
    }

    /// <summary>
    /// Scans a grid of pose offsets around a centre pose and keeps the best.
    /// </summary>
    private static void Search(
        List<Lidar.Measurement> measurements,
        Pos centre,
        Pos lidar_offset,
        float xy_range,
        float xy_step,
        float angle_range,
        float angle_step,
        ref float best_x,
        ref float best_y,
        ref float best_a,
        ref float best_score)
    {
        int xy_steps = Mathf.Max(1, Mathf.RoundToInt(xy_range / xy_step));
        int angle_steps = Mathf.Max(1, Mathf.RoundToInt(angle_range / angle_step));

        Vector2[] buffer = new Vector2[measurements.Count];

        for (int ai = -angle_steps; ai <= angle_steps; ai++)
        {
            float angle = centre.pos_a + ai * angle_step;

            for (int xi = -xy_steps; xi <= xy_steps; xi++)
            {
                float x = centre.pos_x + xi * xy_step;

                for (int yi = -xy_steps; yi <= xy_steps; yi++)
                {
                    float y = centre.pos_y + yi * xy_step;
                    Pos candidate = new Pos { pos_x = x, pos_y = y, pos_a = angle };

                    // Project and score without allocating.
                    float score = ScoreProjected(ProjectScan(measurements, candidate, lidar_offset, buffer));

                    if (score > best_score)
                    {
                        best_score = score;
                        best_x = x;
                        best_y = y;
                        best_a = angle;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Vertex offset of the parabola through (left, centre, right), used to get
    /// sub-cell accuracy from three grid samples.
    /// </summary>
    private static float ParabolaOffset(float left, float centre, float right, float step)
    {
        float denominator = left - 2f * centre + right;

        if (Mathf.Abs(denominator) < Mathf.Epsilon)
        {
            return 0f;
        }

        float offset = 0.5f * (left - right) / denominator;

        // Clamp to half a cell: the vertex should sit between the samples.
        return Mathf.Clamp(offset, -0.5f, 0.5f) * step;
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
