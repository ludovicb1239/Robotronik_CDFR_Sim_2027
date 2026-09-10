using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public struct Pos
{
    public float pos_x; // mm
    public float pos_y; // mm
    public float pos_a; // degs
}

public class Robot : MonoBehaviour
{
    [SerializeField] private Lidar lidar;
    [SerializeField] private Pos lidar_offset;

    [Header("Simulated odometry noise")]
    [SerializeField, Range(0f, 500f)] private float position_noise_mm = 0f;
    [SerializeField, Range(0f, 45f)] private float angle_noise_deg = 0f;

    void Start()
    {
        
    }

    void Update()
    {
        if (lidar == null)
        {
            return;
        }

        if (lidar.UpdateLidar())
        {
            // Frame conversion: Unity yaw turns clockwise about +Y, while the
            // estimator works CCW with 0 deg along +X and 90 deg along +Y.
            // Negating yaw moves into that frame; unity_x is estimator X and
            // unity_z is estimator Y.
            float unity_yaw = transform.eulerAngles.y;

            // A full revolution is done: lidar.measurements is ready to process.
            Pos approximate_position = new Pos
            {
                pos_x = transform.position.x * 1000f + Random.Range(-position_noise_mm, position_noise_mm),
                pos_y = transform.position.z * 1000f + Random.Range(-position_noise_mm, position_noise_mm),
                pos_a = -unity_yaw + Random.Range(-angle_noise_deg, angle_noise_deg),
            };

            Pos estimated_position = EstimatePosition(approximate_position, lidar.measurements, lidar_offset);

            // Ground truth in the estimator's frame, for measuring the residual.
            Pos real_position = new Pos
            {
                pos_x = transform.position.x * 1000f,
                pos_y = transform.position.z * 1000f,
                pos_a = -unity_yaw,
            };

            Debug.Log($"PosEstimator: error " +
                      $"({approximate_position.pos_x - real_position.pos_x:F0}, " +
                      $"{approximate_position.pos_y - real_position.pos_y:F0}, " +
                      $"{approximate_position.pos_a - real_position.pos_a:F1}) " +
                      $"-> residual " +
                      $"({estimated_position.pos_x - real_position.pos_x:F0}, " +
                      $"{estimated_position.pos_y - real_position.pos_y:F0}, " +
                      $"{estimated_position.pos_a - real_position.pos_a:F1})");

            lidar.BeginSweep();
        }
    }

    /// <summary>
    /// Refines the robot's position using the latest lidar scan.
    /// </summary>
    public Pos EstimatePosition(Pos approximate_position, List<Lidar.Measurement> measurements, Pos lidar_offset)
    {
        return PosEstimator.EstimatePosition(approximate_position, measurements, lidar_offset);
    }
}
