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
    private List<Vector3> hit_positions = new List<Vector3>();

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
        measurements.Clear();

        float degrees_per_ray = 360f / Mathf.Max(1, rays_per_scan);

        for (int i = 0; i < rays_per_scan; i++)
        {
            CastRay(i * degrees_per_ray);
        }

        UpdateLineRenderer();
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
}
