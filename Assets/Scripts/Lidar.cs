using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(LineRenderer))]
public class Lidar : MonoBehaviour
{
    [System.Serializable]
    public struct Measurement
    {
        public float angle;    // degrees
        public float distance; // meters
    }

    [SerializeField] private float spinning_frequency_hz = 10f; // Hz
    [SerializeField, Min(1)] private int rays_per_scan = 330;   // rays per full revolution
    [SerializeField] private float sensor_precision_mm = 5f;    // +/- error in mm
    [SerializeField] private LayerMask ignore_hitbox_mask;      // hit, drawn, but not measured
    [SerializeField] private LineRenderer lineRenderer;

    /// <summary>
    /// Draws the rays the obstacle filter discarded, so the classification can be
    /// inspected visually instead of inferred from the surviving ray count. A
    /// second renderer is used rather than recolouring the main one because the
    /// discarded rays are not a subset of the kept ones - they are a different set
    /// of points, drawn over the same scan.
    /// </summary>
    [SerializeField] private LineRenderer bad_line_renderer;

    private List<Vector3> hit_positions = new List<Vector3>();

    /// <summary>
    /// Hit points of the rays the filter rejected, parallel to the discarded
    /// measurements. Drawn by <see cref="bad_line_renderer"/> and cleared with the
    /// rest of the scan.
    /// </summary>
    private List<Vector3> discarded_positions = new List<Vector3>();

    // (angle in degrees, distance in meters) measured this scan
    public List<Measurement> measurements = new List<Measurement>();

    private float scan_accumulator = 0f;

    // Box-Muller produces two independent standard normals per call; the spare
    // is kept here so the second one is not thrown away.
    private bool has_spare_gaussian;
    private float spare_gaussian;

    /// <summary>
    /// Draws a zero-mean Gaussian with the given standard deviation.
    ///
    /// Box-Muller: two uniform draws map to a radius and an angle, and the
    /// resulting point has independent normal components. The cosine and sine
    /// terms are two separate draws, so caching one halves the transcendental
    /// work per scan ray.
    /// </summary>
    private float Gaussian(float standard_deviation)
    {
        if (has_spare_gaussian)
        {
            has_spare_gaussian = false;
            return spare_gaussian * standard_deviation;
        }

        // u1 must stay strictly above zero because the radius takes its log.
        float u1 = 1f - Random.value;
        float u2 = Random.value;

        float radius = Mathf.Sqrt(-2f * Mathf.Log(u1));
        float angle = 2f * Mathf.PI * u2;

        spare_gaussian = radius * Mathf.Sin(angle);
        has_spare_gaussian = true;

        return radius * Mathf.Cos(angle) * standard_deviation;
    }

    private void Awake()
    {
        if (lineRenderer == null)
        {
            lineRenderer = GetComponent<LineRenderer>();
        }

        lineRenderer.useWorldSpace = true;
        lineRenderer.loop = false;

        // The bad renderer is optional: when it is not wired up the filter still
        // runs and the discarded set is still tracked, it is simply not drawn.
        if (bad_line_renderer != null)
        {
            bad_line_renderer.useWorldSpace = true;
            bad_line_renderer.loop = false;
        }
    }

    private void Start()
    {
        BeginSweep();
    }

    /// <summary>Advances the lidar by one frame. Called by the robot controlling it.</summary>
    /// <returns>True when <see cref="measurements"/> has just been filled (one complete scan).</returns>
    public bool UpdateLidar()
    {
        // The lidar spins at a fixed rate but a full scan is produced in a single
        // step: the whole revolution is cast at once rather than accumulated ray
        // by ray, so a scan is instantaneous when it happens.
        scan_accumulator += Time.deltaTime * spinning_frequency_hz;

        if (scan_accumulator < 1f)
        {
            return false;
        }

        scan_accumulator -= 1f;

        CastScan();
        return true;
    }

    public void BeginSweep()
    {
        hit_positions.Clear();
        discarded_positions.Clear();
        measurements.Clear();
        scan_accumulator = 0f;
    }

    /// <summary>
    /// Casts every ray of one full revolution at once. The beams are spread
    /// evenly over 360 degrees, so the scan is a complete, self-consistent
    /// snapshot of the field at this instant.
    ///
    /// Scans are strictly one shot: each revolution is cast in a single step at
    /// a single instant and replaces the previous measurements, so no scan is
    /// ever accumulated from rays captured at different times. That is what
    /// makes the geometric model exact - every ray in a scan shares one robot
    /// pose, and the pose the estimator recovers is the pose at the moment of
    /// the cast. There is no motion distortion and no scan-matching between
    /// successive scans to account for.
    /// </summary>
    private void CastScan()
    {
        hit_positions.Clear();
        discarded_positions.Clear();
        measurements.Clear();

        float degrees_per_ray = 360f / Mathf.Max(1, rays_per_scan);

        for (int i = 0; i < rays_per_scan; i++)
        {
            CastRay(i * degrees_per_ray);
        }

        RemoveOccludedRays();

        UpdateLineRenderer();
        UpdateBadLineRenderer();
    }

    /// <summary>
    /// Drops rays that hit an occluding obstacle rather than a wall, by
    /// classifying the scan in both directions and removing only the rays that
    /// both classifications reject.
    ///
    /// Walking the scan in bearing order, a sudden drop in range means the beam
    /// has just landed on something nearer than whatever it was hitting before -
    /// an obstacle blocking the view. A sudden rise back out means the beam has
    /// cleared that obstacle's edge. The run of rays between the two is the
    /// occluded span, and it covers both the obstacle's own front face and the
    /// shadow it casts behind itself.
    ///
    /// The difficulty is deciding which side of a jump is the occluder and which
    /// is the wall, because that depends on the scan's direction of travel. Walk
    /// the rays forwards and the run that follows a drop looks occluded; walk
    /// them backwards and the run that precedes that same drop looks occluded
    /// instead. A range-only rule cannot tell a real obstacle edge from a convex
    /// corner of the field itself, where the range legitimately drops with
    /// nothing occluding it, so a forwards-only rule can eat good corner
    /// geometry - exactly the features that pin down the pose.
    ///
    /// Requiring both directions to agree removes that ambiguity. A ray is
    /// discarded only when the forwards pass and the backwards pass both place it
    /// inside an occluded span. At a concave obstacle, both passes agree and it is
    /// removed. At a convex corner, the two passes disagree, so the geometry is
    /// kept and the filter stays conservative - it drops less than a
    /// single-direction rule, and what it does drop is much less likely to be
    /// genuine wall.
    ///
    /// Applied once per scan, before the line renderer runs, so what is drawn and
    /// what is reported always agree. Thresholds are hardcoded for now. JUMP_MM
    /// is well above the sensor's own 5 mm range noise, so noise on a flat wall
    /// is not mistaken for an edge, while a real obstacle edge moves the range by
    /// hundreds of mm.
    /// </summary>
    private void RemoveOccludedRays()
    {
        const float JUMP_MM = 50f;

        int count = measurements.Count;

        if (count < 3)
        {
            return;
        }

        // Work in millimetres, since the tolerance is specified that way.
        float[] ranges_mm = new float[count];

        for (int i = 0; i < count; i++)
        {
            ranges_mm[i] = measurements[i].distance * 1000f;
        }

        // The scan is a closed revolution, so ray count-1 is adjacent to ray 0 in
        // the world. Walking the array as a plain line therefore misses any
        // occluded span that straddles the seam, which is exactly the case where
        // the first and last rays of the scan are on an obstacle.
        //
        // The passes below consequently wrap around the ring. Each walks the
        // cycle once starting from a seed ray assumed unoccluded, then continues
        // a full lap so every ray is visited after a boundary decision has been
        // made at least once. The seed is the longest-range ray: it is the one
        // worst placed to be inside an occluded span, since an occluder can only
        // bring a range down.
        int seed = 0;

        for (int i = 1; i < count; i++)
        {
            if (ranges_mm[i] > ranges_mm[seed])
            {
                seed = i;
            }
        }

        // Forward pass: a ray is occluded when the nearest preceding sharp move
        // was a drop rather than a rise. Starting at the seed and going a full
        // lap, plus two steps, guarantees the first ray visited after the seed
        // has inherited a real boundary decision rather than an arbitrary false.
        bool[] occluded_forward = new bool[count];
        bool occluded = false;

        for (int step = 1; step <= count + 1; step++)
        {
            int i = (seed + step) % count;
            int previous = (seed + step - 1) % count;

            float delta = ranges_mm[i] - ranges_mm[previous];

            if (delta < -JUMP_MM)
            {
                occluded = true;
            }
            else if (delta > JUMP_MM)
            {
                occluded = false;
            }

            // The wrap-around steps only exist to settle the state at the seam;
            // they must not overwrite an already-final classification.
            if (step <= count)
            {
                occluded_forward[i] = occluded;
            }
        }

        // Backward pass: the same walk in the opposite direction. Starting from
        // the other end makes each span's boundary the opposite kind of jump, so
        // the two passes disagree wherever a jump is not a real occlusion.
        bool[] occluded_backward = new bool[count];
        occluded = false;

        for (int step = 1; step <= count + 1; step++)
        {
            int i = (seed - step + count * 2) % count;
            int previous = (seed - step + 1 + count * 2) % count;

            float delta = ranges_mm[i] - ranges_mm[previous];

            if (delta < -JUMP_MM)
            {
                occluded = true;
            }
            else if (delta > JUMP_MM)
            {
                occluded = false;
            }

            if (step <= count)
            {
                occluded_backward[i] = occluded;
            }
        }

        // Keep a ray unless both passes agree that it is occluded. Rejected rays
        // are copied out to their own list first, so the bad renderer can draw
        // exactly the geometry the filter removed.
        bool[] keep = new bool[count];

        for (int i = 0; i < count; i++)
        {
            keep[i] = !(occluded_forward[i] && occluded_backward[i]);
        }

        for (int i = 0; i < count; i++)
        {
            if (!keep[i] && i < hit_positions.Count)
            {
                discarded_positions.Add(hit_positions[i]);
            }
        }

        // Rebuild both parallel lists from the surviving rays, so the drawn
        // points and the reported measurements cannot drift apart.
        int write = 0;

        for (int i = 0; i < count; i++)
        {
            if (!keep[i])
            {
                continue;
            }

            if (write != i)
            {
                measurements[write] = measurements[i];

                if (i < hit_positions.Count)
                {
                    hit_positions[write] = hit_positions[i];
                }
            }

            write++;
        }

        if (write < measurements.Count)
        {
            measurements.RemoveRange(write, measurements.Count - write);
        }

        if (write < hit_positions.Count)
        {
            hit_positions.RemoveRange(write, hit_positions.Count - write);
        }
    }

    private void CastRay(float angle_degrees)
    {
        Vector3 origin = transform.position;
        Quaternion spin = Quaternion.Euler(0f, angle_degrees, 0f);
        Vector3 direction = transform.parent != null
            ? transform.parent.TransformDirection(spin * Vector3.forward)
            : spin * transform.forward;

        float distance = 100f;
        bool ignored = false;

        if (Physics.Raycast(origin, direction, out RaycastHit hit, 100f))
        {
            distance = hit.distance;
            ignored = (ignore_hitbox_mask.value & (1 << hit.collider.gameObject.layer)) != 0;
        }

        // Randomise the measured distance with Gaussian noise whose standard
        // deviation is sensor_precision_mm. A uniform draw over
        // +/- sensor_precision_mm has standard deviation precision/sqrt(3), so
        // it would misrepresent a real sensor: distance noise is the sum of
        // many small independent effects and is therefore bell shaped, not
        // flat. Matching the estimator's Gaussian likelihood to the noise the
        // simulator actually injects keeps the two consistent and stops the
        // model from being systematically over- or under-confident.
        //
        // Gaussian tails are unbounded, so the result is clamped to the
        // physically sensible range instead of allowing a negative range.
        float noise_m = Gaussian(sensor_precision_mm) * 0.001f;
        float measured_distance = Mathf.Clamp(distance + noise_m, 0f, 100f);

        // Ignored hitboxes are drawn but never reported as a measurement.
        if (!ignored)
        {        
            hit_positions.Add(origin + direction * measured_distance);

            // Reported in the consumer's frame: X forward, Y left, angles CCW,
            // which is the opposite sense to Unity's yaw about +Y.
            measurements.Add(new Measurement { angle = -angle_degrees, distance = measured_distance });
        }
    }

    private void UpdateLineRenderer()
    {
        if (hit_positions.Count < 2)
        {
            return;
        }

        lineRenderer.positionCount = hit_positions.Count*2;

        for (int i = 0; i < hit_positions.Count; i++)
        {
            lineRenderer.SetPosition(i*2, transform.position);
            lineRenderer.SetPosition(i*2+1, hit_positions[i]);
        }
    }

    /// <summary>
    /// Draws the rays the obstacle filter discarded, from the sensor out to the
    /// rejected hit point, using the same paired-position layout as the main
    /// renderer.
    ///
    /// Handles the empty case explicitly: a scan in which the filter rejected
    /// nothing must clear the previous frame's lines, or the bad renderer would
    /// keep showing geometry that is no longer being discarded. The count is set
    /// to zero rather than left alone, so the stale positions are dropped.
    /// </summary>
    private void UpdateBadLineRenderer()
    {
        if (bad_line_renderer == null)
        {
            return;
        }

        if (discarded_positions.Count == 0)
        {
            bad_line_renderer.positionCount = 0;
            return;
        }

        bad_line_renderer.positionCount = discarded_positions.Count * 2;

        for (int i = 0; i < discarded_positions.Count; i++)
        {
            bad_line_renderer.SetPosition(i * 2, transform.position);
            bad_line_renderer.SetPosition(i * 2 + 1, discarded_positions[i]);
        }
    }
}
