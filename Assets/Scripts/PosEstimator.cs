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
/// Scoring: each ray is resolved along its own bearing against the field walls.
/// The wall's range is found by exact ray intersection, and the measured range
/// is compared against it. A measurement that is shorter than the wall's range
/// was stopped early by an obstacle, which is a valid and expected outcome:
/// walls are the furthest thing the robot can see, so such a ray is scored at
/// the floor and contributes nothing to the mean. This is what keeps obstacle
/// geometry from pulling the pose, and it needs no tolerance on position, no
/// clustering and no pose-dependent classification of which rays are "far".
///
/// The remaining rays - the ones that did reach a wall - are scored by a robust
/// kernel of their range error, and the pose score is the mean over all rays.
/// The kernel is bounded so no single ray can drive the score, and it never
/// becomes exactly flat so the search always has a gradient to follow.
///
/// Algorithm: coordinate-descent scan matching at a fixed stride. Instead of
/// scoring the whole (dx, dy, da) box, the search sweeps one axis at a time -
/// heading, then X, then Y - because a robot pose error is dominated by a single
/// axis at a time and each axis converges independently around a tight odometry
/// prior. Each axis is sampled at TARGET_STEP_MM spacing, so the span a sweep
/// covers is set by the number of samples rather than the other way round. This
/// separation matters: the span must stay wide enough to contain the error, while
/// the spacing must be fine enough to resolve the peak, and tying the two
/// together makes a wide search necessarily coarse. Sweeping repeats until a
/// whole sweep stops improving the score.
///
/// Cost: a box search samples (2na+1)(2nx+1)(2ny+1) full poses, while one sweep
/// samples only (2na+1)+(2nx+1)+(2ny+1) and that count is fixed by
/// POINTS_PER_SWEEP regardless of how wide or fine the stride is. Every
/// candidate is scored as a full pose, because the occlusion test depends on
/// which wall each bearing reaches and that is not a translation of the previous
/// answer.
///
/// The inner loop is kept cheap by three things, in order of impact:
/// 1. the field has only eight boundary segments, so an exact ray intersection is
///    a handful of arithmetic per segment and needs no spatial index;
/// 2. each ray's bearing is baked once per scan, so a candidate pose costs a
///    rotation of the precomputed bearing rather than fresh trigonometry;
/// 3. the score is a bounded rational kernel, which the compiler turns into a
///    multiply-add, a divide and no transcendental call at all.
/// </summary>
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

    /// <summary>
    /// Fills the flattened segment tables from <see cref="FIELD_OUTLINE"/>. The
    /// polygon's last vertex is wired back to the first, so the closing edge is
    /// present in the table like any other and the raycast needs no wrap-around
    /// special case.
    /// </summary>
    static PosEstimator()
    {
        int count = FIELD_OUTLINE.Length;

        segment_start = new Vector2[count];
        segment_edge = new Vector2[count];

        for (int i = 0; i < count; i++)
        {
            Vector2 a = FIELD_OUTLINE[i];
            Vector2 b = FIELD_OUTLINE[(i + 1) % count];

            segment_start[i] = a;
            segment_edge[i] = b - a;
        }
    }

    // Per-ray invariants for the current scan, baked by PrepareScan.
    private static float[] scan_cos;
    private static float[] scan_sin;
    private static float[] scan_range_mm;

    // --- Search schedule ---------------------------------------------------
    /// <summary>
    /// Sample points taken on each side of the current best, per sweep. A sweep
    /// therefore evaluates 2*POINTS_PER_SWEEP+1 candidates per axis, and covers a
    /// span of 2*POINTS_PER_SWEEP strides.
    /// </summary>
    private const int POINTS_PER_SWEEP = 20;

    /// <summary>
    /// Hard ceiling on the number of coordinate-descent sweeps per estimate.
    ///
    /// This is a ceiling, not a schedule: the search normally exits early once a
    /// whole sweep stops improving, and the ceiling only exists so a scan that is
    /// still crawling cannot run forever.
    ///
    /// It used to be 12 with no early exit, which measurement showed was the
    /// binding constraint rather than the available range. Across 1036 recorded
    /// scans, the scans that failed the 5 mm bar still improved the score on
    /// EVERY sweep they were given - median 12 of 12 improving - while clamped
    /// sweeps were rare (median 1). That combination means the search was not
    /// stuck and was not running out of range: it was converging geometrically
    /// along one axis at a time and being cut off mid-crawl. Since coordinate
    /// descent interleaves the axes, a large correction on a single axis needs
    /// many sweeps to accumulate, and 12 was simply not enough of them.
    /// </summary>
    private const int MAX_TOTAL_STEPS = 40;

    /// <summary>
    /// Absolute score gain below which a whole sweep counts as having converged.
    ///
    /// Score gains fall off geometrically as the pose approaches the peak, so a
    /// sweep that moves the score by less than this has nothing left to find.
    /// This is what lets the easy majority of scans exit after a handful of
    /// sweeps instead of paying the full ceiling.
    /// </summary>
    private const float SWEEP_CONVERGENCE_EPSILON = 1e-6f;

    /// <summary>
    /// Target spacing between adjacent samples along an axis, in mm.
    ///
    /// This is what actually sets the achievable precision, and decoupling it
    /// from the search range is the point. A sweep samples
    /// 2*POINTS_PER_SWEEP+1 points spanning the current range, so its spacing is
    /// range / POINTS_PER_SWEEP. If the range alone decided the spacing, a wide
    /// search would be coarse and only a narrow search would resolve the peak,
    /// which forces a trade: the range must stay wide enough to contain the
    /// error while the spacing must become fine enough to pin the peak down.
    ///
    /// The sweep therefore takes a fixed number of strides of this size,
    /// centred on the current best, instead of a fixed span divided evenly. The
    /// span it covers is 2*POINTS_PER_SWEEP*this value, which at 20 points is
    /// +/- 60 mm - comfortably more than the prior's +/75 mm error once the
    /// first sweep has moved the pose most of the way, and fine enough that the
    /// residual is set by the objective rather than by the grid.
    /// </summary>
    private const float TARGET_STEP_MM = 3f;

    /// <summary>
    /// Target angular spacing between adjacent samples, in degrees. The angular
    /// equivalent of <see cref="TARGET_STEP_MM"/>; the span covered is
    /// 2*POINTS_PER_SWEEP*this value, i.e. about +/- 2.4 deg.
    ///
    /// A 5 deg prior error is 175 mm of arc at 2 m, so the first sweep has to
    /// cover a wide angle; later sweeps only need to resolve the peak, and this
    /// spacing is fine enough for that at any range in the field.
    /// </summary>
    private const float TARGET_ANGLE_STEP_DEG = 0.12f;

    /// <summary>
    /// Range slack, in mm, below which a measured range counts as a wall hit
    /// rather than an occlusion.
    ///
    /// A ray is occluded when it was stopped before reaching the wall, which is a
    /// purely one-sided comparison: walls are the furthest thing the robot can
    /// see, so a measurement can only ever be shorter than the wall's range, never
    /// longer. The slack exists to absorb the sensor's own range noise and the
    /// candidate pose's error, both of which make the measured and predicted
    /// ranges differ even for a genuine wall hit.
    /// </summary>
    private const float OCCLUSION_SLACK_MM = 3f * SCORE_SIGMA_MM;

    /// <summary>
    /// Target angular spacing between adjacent samples in the refinement phase,
    /// in degrees. The angular counterpart of <see cref="REFINE_STEP_MM"/>.
    /// </summary>
    private const float REFINE_ANGLE_STEP_DEG = 0.02f;

    /// <summary>
    /// Final spacing between adjacent samples, in mm, used by the refinement
    /// phase after the coarse phase has localised the peak.
    ///
    /// The coarse stride sets the span a sweep can travel, but it also sets the
    /// finest distinction the search can make: a coordinate-descent step can only
    /// land on a sample, so the reachable poses form a grid of the stride's
    /// spacing. Measurement showed exactly that limit. Across 400 scans the mean
    /// residual was 0.26 mm - so there was no bias at all - yet the scatter was
    /// about 6 mm, roughly two coarse strides, and the search always stopped
    /// after two to five sweeps because no sampled candidate improved the pose.
    /// That is the signature of a search that has run out of grid, not of one
    /// that has run out of information: the truth lay between samples.
    ///
    /// The refinement phase re-sweeps at this spacing over a much smaller span,
    /// reached by taking POINTS_PER_REFINEMENT_SWEEP strides, so it can resolve
    /// a peak the coarse grid straddled without paying for a fine grid across the
    /// whole search area.
    /// </summary>
    private const float REFINE_STEP_MM = 1f;

    /// <summary>
    /// Sample points taken on each side of the current best, per refinement
    /// sweep. Smaller than POINTS_PER_SWEEP because a refinement sweep only has
    /// to cover the coarse grid's spacing, not the whole prior error.
    /// </summary>
    private const int POINTS_PER_REFINEMENT_SWEEP = 8;

    /// <summary>
    /// Sweeps allowed in the refinement phase. Coordinate descent needs a few
    /// passes for the axes to settle once the steps are this small, and the
    /// phase exits early as soon as a sweep stops improving.
    /// </summary>
    private const int MAX_REFINEMENT_SWEEPS = 8;

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

    /// <summary>Score of the approximate (prior) pose, at the start of the last estimate.</summary>
    public static float LastBaseScore { get; private set; }

    /// <summary>
    /// Score of the pose the search returned. Compared against the score at the
    /// true pose this says whether the search stopped early or the objective
    /// itself is biased.
    /// </summary>
    public static float LastBestScore { get; private set; }

    /// <summary>
    /// Final search half-width in mm, after the last zoom. When this is small
    /// while a residual remains, the range shrank past the remaining correction
    /// and the search could no longer travel to the peak.
    /// </summary>
    public static float LastFinalRangeMm { get; private set; }

    /// <summary>Final angular search half-width in degrees, after the last zoom.</summary>
    public static float LastFinalAngleRangeDeg { get; private set; }

    /// <summary>
    /// Sweeps in the last estimate in which at least one axis picked the extreme
    /// sample it was offered. Picking the edge means the true optimum lay outside
    /// the sampled window, so the zoom shrank the range before the axis had
    /// finished travelling - the direct symptom of a schedule that zooms too
    /// eagerly.
    /// </summary>
    public static int LastClampedSweeps { get; private set; }

    /// <summary>
    /// Sweeps in the last estimate that improved the score at all. If this is
    /// well below the sweep count, the search was flat for the later sweeps,
    /// which points at the objective rather than the schedule.
    /// </summary>
    public static int LastImprovingSweeps { get; private set; }

    /// <summary>
    /// Rays in the last estimate whose measured range matched the wall's range on
    /// the same bearing, i.e. rays that were genuine wall hits rather than
    /// occlusions. Reported for diagnosis: a correct pose should classify most
    /// rays as wall hits, whereas a pose that is sliding along a wall tends to
    /// leave many rays unexplained.
    /// </summary>
    public static int LastWallHitCount { get; private set; }

    /// <summary>
    /// Distance from a ray origin along a direction to the first field wall, in
    /// mm, or <see cref="float.PositiveInfinity"/> when the ray never meets one.
    ///
    /// Standard segment intersection in the ray's frame: the wall from A along E
    /// meets the ray from O along D where the two cross, which is a 2x2 solve.
    /// A crossing counts only when it lies ahead of the ray (t &gt; 0) and inside
    /// the finite wall segment (0 &lt;= u &lt;= 1), so the result is the distance
    /// to the wall itself rather than to the infinite line it lies on.
    ///
    /// This is the piece that makes an occlusion test possible at all. Comparing
    /// a measured range against the wall's range along the same bearing is a
    /// like-for-like comparison of two distances along one ray; measuring the
    /// distance from a hit point to the nearest wall, which is what the score did
    /// before, mixes a measurement with a quantity that is not what the sensor
    /// reported.
    /// </summary>
    private static float RaycastWall(float origin_x, float origin_y, float dir_x, float dir_y)
    {
        float nearest = float.PositiveInfinity;

        for (int i = 0; i < segment_start.Length; i++)
        {
            float ex = segment_edge[i].x;
            float ey = segment_edge[i].y;

            // Denominator of the 2x2 solve. Zero means ray and wall are parallel.
            float den = dir_x * ey - dir_y * ex;

            if (Mathf.Abs(den) < 1e-9f)
            {
                continue;
            }

            float ax = segment_start[i].x - origin_x;
            float ay = segment_start[i].y - origin_y;

            float t = (ax * ey - ay * ex) / den;
            float u = (ax * dir_y - ay * dir_x) / den;

            if (t > 1e-6f && u >= 0f && u <= 1f && t < nearest)
            {
                nearest = t;
            }
        }

        return nearest;
    }

    /// <summary>
    /// Scores an arbitrary pose with the same objective the search maximises.
    ///
    /// This exists purely for diagnosis: comparing the score at the pose the
    /// search returned against the score at the true pose separates the two
    /// possible reasons a search can come up short. If the true pose scores
    /// higher, the search stopped early and the schedule is at fault. If the
    /// returned pose scores higher, the search found the best available peak and
    /// the objective itself is biased, which no amount of search tuning fixes.
    ///
    /// Uses a throwaway workspace so it cannot disturb an estimate in progress,
    /// and it does not touch the Last* diagnostics.
    /// </summary>
    public static float ScoreAt(
        List<Lidar.Measurement> measurements,
        Pos pose,
        Pos lidar_offset)
    {
        PrepareScan(measurements);

        SearchWorkspace workspace = new SearchWorkspace();

        return ScorePose(measurements, pose, lidar_offset, workspace);
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
        LastBaseScore = 0f;
        LastBestScore = 0f;
        LastFinalRangeMm = 0f;
        LastFinalAngleRangeDeg = 0f;
        LastClampedSweeps = 0;
        LastImprovingSweeps = 0;

        if (measurements.Count < 2)
        {
            Reject("scan produced fewer than 2 points");
            return approximate_position;
        }

        PrepareScan(measurements);

        SearchWorkspace workspace = new SearchWorkspace();

        System.Diagnostics.Stopwatch clock = System.Diagnostics.Stopwatch.StartNew();

        float base_score = ScorePose(measurements, approximate_position, lidar_offset, workspace);

        // --- Zooming coordinate-descent search ------------------------------
        PoseEstimate best = Search(measurements, approximate_position, lidar_offset, workspace);

        clock.Stop();
        LastEstimateMs = clock.Elapsed.TotalMilliseconds;
        LastSweepCount = workspace.sweeps_run;
        LastBaseScore = base_score;

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
    /// Scores the scan from a candidate robot pose.
    ///
    /// The lidar origin is placed in the field for the candidate pose, and each
    /// ray is resolved along its own bearing: either it reaches the wall, in
    /// which case the measured range is compared against the wall's range, or it
    /// is stopped short by an obstacle, in which case it is allowed to be shorter
    /// and contributes a constant.
    /// </summary>
    private static float ScorePose(
        List<Lidar.Measurement> measurements,
        Pos pose,
        Pos lidar_offset,
        SearchWorkspace workspace)
    {
        float a_rad = pose.pos_a * Mathf.Deg2Rad;
        float cos_a = Mathf.Cos(a_rad);
        float sin_a = Mathf.Sin(a_rad);

        float origin_x = pose.pos_x + lidar_offset.pos_x * cos_a - lidar_offset.pos_y * sin_a;
        float origin_y = pose.pos_y + lidar_offset.pos_x * sin_a + lidar_offset.pos_y * cos_a;

        float score = 0f;
        int wall_hits = 0;

        for (int i = 0; i < measurements.Count; i++)
        {
            float local_x = scan_cos[i];
            float local_y = scan_sin[i];

            // Ray direction in field coordinates.
            float dir_x = local_x * cos_a - local_y * sin_a;
            float dir_y = local_x * sin_a + local_y * cos_a;

            float wall_range_mm = RaycastWall(origin_x, origin_y, dir_x, dir_y);
            float measured_mm = scan_range_mm[i];

            // Walls are the furthest thing visible, so a measurement shorter than
            // the wall's range means something blocked the beam. Such a ray is
            // uninformative about the pose: it is scored at the floor and adds
            // nothing to the mean, so obstacle geometry can neither attract the
            // pose nor be mistaken for a wall.
            //
            // The comparison is one-sided by construction. A measurement longer
            // than the wall's range is physically impossible; it is treated as a
            // miss and scored at the floor rather than trusted.
            if (measured_mm < wall_range_mm - OCCLUSION_SLACK_MM)
            {
                score += PointScore(OCCLUSION_SLACK_MM * 2f);
                continue;
            }

            float range_error = measured_mm - wall_range_mm;
            score += PointScore(range_error);
            wall_hits++;
        }

        LastWallHitCount = wall_hits;

        return measurements.Count > 0 ? score / measurements.Count : 0f;
    }


    /// <summary>Outcome of one search schedule.</summary>
    private struct PoseEstimate
    {
        public float x;
        public float y;
        public float a;
        public float score;
    }

    /// <summary>Scratch state for one schedule.</summary>
    private sealed class SearchWorkspace
    {
        /// <summary>Sweeps the last search actually ran, for diagnostics.</summary>
        public int sweeps_run;

        /// <summary>
        /// Set by a sweep when the winning candidate was the outermost sample it
        /// was offered, meaning the peak lies beyond the sampled window. Cleared
        /// once per sweep by the caller.
        /// </summary>
        public bool hit_extreme;
    }

    /// <summary>
    /// Runs a coordinate-descent search that sweeps the heading, then X, then Y,
    /// sampling POINTS_PER_SWEEP strides either side of the current best on each
    /// axis at a FIXED stride length.
    ///
    /// The stride is deliberately not derived from a shrinking search range. An
    /// earlier version shrank both together, which coupled two things that need
    /// opposite treatment: the span must stay wide enough to contain the error,
    /// and the spacing must become fine enough to locate the peak. Coupling them
    /// meant a wide search was necessarily coarse, so the pose could only be
    /// pinned down after the range had already collapsed - and if the remaining
    /// error exceeded the collapsed range, the search could never recover it.
    /// Measurement showed exactly that: failing scans improved the score on every
    /// sweep they were given, so they were still converging when the budget ran
    /// out, and their residuals clustered tens of mm from the truth.
    ///
    /// Taking a fixed stride instead makes each sweep cover
    /// 2*POINTS_PER_SWEEP*TARGET_STEP_MM, about +/- 60 mm of translation and
    /// +/- 2.4 deg of heading, with spacing fine enough that the residual is
    /// limited by the objective rather than by the sample grid. A sweep costs the
    /// same either way - it is a fixed number of candidate poses - so this is
    /// strictly a better use of the same budget.
    ///
    /// The loop stops once a whole sweep raises the score by less than
    /// SWEEP_CONVERGENCE_EPSILON, and is otherwise capped at MAX_TOTAL_STEPS. The
    /// cap only binds on scans that are still crawling.
    ///
    /// A second phase then re-sweeps at REFINE_STEP_MM over a narrower span. The
    /// coarse phase cannot resolve finer than its own stride, because a
    /// coordinate-descent step can only land on a sample, so a peak lying between
    /// two coarse samples is unreachable no matter how many coarse sweeps run.
    /// The refinement span is POINTS_PER_REFINEMENT_SWEEP strides, deliberately
    /// wider than one coarse stride so it can reach a peak the coarse grid
    /// straddled, and fine enough that the residual is no longer set by sampling.
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

        int clamped_sweeps = 0;
        int improving_sweeps = 0;

        for (int step_index = 0; step_index < MAX_TOTAL_STEPS; step_index++)
        {
            float score_before = best.score;

            // Each sweep records whether it settled on the outermost sample it
            // was offered, which means the peak lay outside the window.
            workspace.hit_extreme = false;

            SweepAngle(measurements, lidar_offset, workspace, POINTS_PER_SWEEP,
                       TARGET_ANGLE_STEP_DEG, ref best);
            SweepX(measurements, lidar_offset, workspace, POINTS_PER_SWEEP,
                   TARGET_STEP_MM, ref best);
            SweepY(measurements, lidar_offset, workspace, POINTS_PER_SWEEP,
                   TARGET_STEP_MM, ref best);

            if (workspace.hit_extreme)
            {
                clamped_sweeps++;
            }

            float gain = best.score - score_before;

            if (gain > 0f)
            {
                improving_sweeps++;
            }

            workspace.sweeps_run = step_index + 1;

            // Nothing left to find: stop paying for sweeps.
            if (gain < SWEEP_CONVERGENCE_EPSILON)
            {
                break;
            }
        }

        // --- Refinement phase ------------------------------------------------
        // The coarse phase has localised the peak to within about one stride, but
        // a step can only land on a sample, so it cannot resolve finer than the
        // stride's spacing. This phase re-sweeps at REFINE_STEP_MM over a span of
        // POINTS_PER_REFINEMENT_SWEEP strides, which is wide enough to cover the
        // coarse grid spacing and fine enough to resolve the peak inside it.
        //
        // The phase is entered unconditionally: a pose that already sits on the
        // coarse grid still has no way to know whether it is at the peak or one
        // stride short of it, and the first refinement sweep answers that. It
        // exits as soon as a sweep stops improving, so a pose that was already
        // optimal costs only a single sweep.
        int refinement_sweeps = 0;

        for (int step_index = 0; step_index < MAX_REFINEMENT_SWEEPS; step_index++)
        {
            float score_before = best.score;

            SweepAngle(measurements, lidar_offset, workspace, POINTS_PER_REFINEMENT_SWEEP,
                       REFINE_ANGLE_STEP_DEG, ref best);
            SweepX(measurements, lidar_offset, workspace, POINTS_PER_REFINEMENT_SWEEP,
                   REFINE_STEP_MM, ref best);
            SweepY(measurements, lidar_offset, workspace, POINTS_PER_REFINEMENT_SWEEP,
                   REFINE_STEP_MM, ref best);

            refinement_sweeps++;

            float gain = best.score - score_before;

            if (gain > 0f)
            {
                improving_sweeps++;
            }

            workspace.sweeps_run = MAX_TOTAL_STEPS + refinement_sweeps;

            if (gain < SWEEP_CONVERGENCE_EPSILON)
            {
                break;
            }
        }

        LastBestScore = best.score;
        LastFinalRangeMm = POINTS_PER_REFINEMENT_SWEEP * REFINE_STEP_MM;
        LastFinalAngleRangeDeg = POINTS_PER_REFINEMENT_SWEEP * REFINE_ANGLE_STEP_DEG;
        LastClampedSweeps = clamped_sweeps;
        LastImprovingSweeps = improving_sweeps;

        return best;
    }

    /// <summary>
    /// Sweeps X with Y and the heading fixed.
    ///
    /// Each candidate is scored as a full pose rather than by translating cached
    /// projected points. The occlusion test compares a measured range against the
    /// wall's range along the same bearing, and moving the robot parallel to a
    /// wall can bring a different wall into view along that bearing, so the two
    /// sides of the comparison are no longer related by a pure translation. The
    /// cheaper translation shortcut is therefore no longer valid.
    /// </summary>
    private static void SweepX(
        List<Lidar.Measurement> measurements,
        Pos lidar_offset,
        SearchWorkspace workspace,
        int steps,
        float step,
        ref PoseEstimate best)
    {
        float best_x = best.x;
        float best_score = best.score;
        int best_index = 0;

        for (int xi = -steps; xi <= steps; xi++)
        {
            float x = best.x + xi * step;

            Pos candidate = new Pos { pos_x = x, pos_y = best.y, pos_a = best.a };
            float score = ScorePose(measurements, candidate, lidar_offset, workspace);

            if (score > best_score)
            {
                best_score = score;
                best_x = x;
                best_index = xi;
            }
        }

        best.x = best_x;
        best.score = best_score;

        // The extreme sample winning means the true optimum is outside the window
        // on this axis, so the zoom must not be allowed to strand it.
        if (best_index == -steps || best_index == steps)
        {
            workspace.hit_extreme = true;
        }
    }

    /// <summary>
    /// Sweeps Y with X and the heading fixed, scoring each candidate as a full
    /// pose for the same reason as the X sweep: the occlusion test depends on
    /// which wall each bearing reaches, and that is not a pure translation of the
    /// previous answer.
    /// </summary>
    private static void SweepY(
        List<Lidar.Measurement> measurements,
        Pos lidar_offset,
        SearchWorkspace workspace,
        int steps,
        float step,
        ref PoseEstimate best)
    {
        float best_y = best.y;
        float best_score = best.score;
        int best_index = 0;

        for (int yi = -steps; yi <= steps; yi++)
        {
            float y = best.y + yi * step;

            Pos candidate = new Pos { pos_x = best.x, pos_y = y, pos_a = best.a };
            float score = ScorePose(measurements, candidate, lidar_offset, workspace);

            if (score > best_score)
            {
                best_score = score;
                best_y = y;
                best_index = yi;
            }
        }

        best.y = best_y;
        best.score = best_score;

        if (best_index == -steps || best_index == steps)
        {
            workspace.hit_extreme = true;
        }
    }

    /// <summary>
    /// Sweeps the heading with X and Y fixed, scoring each candidate as a full
    /// pose. Rotation swings every ray, so both sides of the occlusion comparison
    /// change together and the test has to be re-evaluated per candidate.
    /// </summary>
    private static void SweepAngle(
        List<Lidar.Measurement> measurements,
        Pos lidar_offset,
        SearchWorkspace workspace,
        int steps,
        float step,
        ref PoseEstimate best)
    {
        float best_a = best.a;
        float best_score = best.score;
        int best_index = 0;

        for (int ai = -steps; ai <= steps; ai++)
        {
            float angle = best.a + ai * step;

            Pos candidate = new Pos { pos_x = best.x, pos_y = best.y, pos_a = angle };
            float score = ScorePose(measurements, candidate, lidar_offset, workspace);

            if (score > best_score)
            {
                best_score = score;
                best_a = angle;
                best_index = ai;
            }
        }

        best.a = best_a;
        best.score = best_score;

        if (best_index == -steps || best_index == steps)
        {
            workspace.hit_extreme = true;
        }
    }

    /// <summary>
    /// Score contribution of a ray, as a Geman-McClure robust kernel of its
    /// range error:
    ///
    ///     s(e) = rolloff^2 / (rolloff^2 + 2 * e^2)
    ///
    /// where the error is the measured range minus the wall's range along the
    /// same bearing. The kernel is 1 on a perfect match, halving at about
    /// 0.64 * rolloff, so a ray that lands on a wall carries strong positional
    /// information. Its tails decay only quadratically, so the function is
    /// strictly decreasing everywhere with no flat region: every ray keeps a
    /// direction to improve, which is what lets a pose whose wall rays all start
    /// far away still be pulled in.
    ///
    /// Being bounded also means no single ray can dominate the mean, which is why
    /// a robust kernel is used rather than an unweighted squared error. Rays that
    /// were occluded do not come through here at all: they are scored at the
    /// floor by <see cref="ScorePose"/>.
    /// </summary>
    private static float PointScore(float range_error)
    {
        float rolloff_squared = SCORE_ROLLOFF_MM * SCORE_ROLLOFF_MM;
        return rolloff_squared / (rolloff_squared + 2f * range_error * range_error);
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
