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
    [SerializeField] private float ray_frequency_hz = 100f;     // Hz
    [SerializeField] private float sensor_precision_mm = 5f;    // +/- error in mm
    [SerializeField] private LayerMask ignore_hitbox_mask;      // hit, drawn, but not measured
    [SerializeField] private LineRenderer lineRenderer;
    private List<Vector3> hit_positions = new List<Vector3>();

    // (angle in degrees, distance in meters) measured this revolution
    public List<Measurement> measurements = new List<Measurement>();

    private float current_angle = 0f;
    private float degrees_per_ray = 0f;
    private float ray_accumulator = 0f;

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
    /// <returns>True when <see cref="measurements"/> has just been filled (end of a revolution).</returns>
    public bool UpdateLidar()
    {
        bool revolution_complete = false;

        ray_accumulator += Time.deltaTime * ray_frequency_hz;

        while (ray_accumulator >= 1f)
        {
            ray_accumulator -= 1f;

            current_angle += degrees_per_ray;

            if (current_angle >= 360f)
            {
                current_angle -= 360f;
                revolution_complete = true;
            }

            CastRay();
        }

        transform.localRotation = Quaternion.Euler(0f, current_angle, 0f);
        UpdateLineRenderer();

        return revolution_complete;
    }

    public void BeginSweep()
    {
        float rays_per_revolution = Mathf.Max(1f, ray_frequency_hz / spinning_frequency_hz);
        degrees_per_ray = 360f / rays_per_revolution;

        hit_positions.Clear();
        measurements.Clear();
        current_angle = 0f;
        ray_accumulator = 0f;
    }

    private void CastRay()
    {
        Vector3 origin = transform.position;
        Quaternion spin = Quaternion.Euler(0f, current_angle, 0f);
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

        // Randomise the measured distance by +/- sensor_precision_mm.
        float noise_m = Random.Range(-sensor_precision_mm, sensor_precision_mm) * 0.001f;
        float measured_distance = Mathf.Clamp(distance + noise_m, 0f, 100f);

        // Ignored hitboxes are drawn but never reported as a measurement.
        if (!ignored)
        {        
            hit_positions.Add(origin + direction * measured_distance);

            // Reported in the consumer's frame: X forward, Y left, angles CCW,
            // which is the opposite sense to Unity's yaw about +Y.
            measurements.Add(new Measurement { angle = -current_angle, distance = measured_distance });
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
