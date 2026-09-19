using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;

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

    [Header("Controls")]
    [SerializeField] private float movement_speed = 2f;
    [SerializeField] private float rotation_speed = 90f;

    [Header("HUD")]
    [SerializeField] private Text positionnement_text;

    private InputAction move_action;
    private InputAction rotate_action;

    private int positionnement_total;
    private int positionnement_fails;

    private float residual_x_mm;
    private float residual_y_mm;
    private float residual_a_deg;

    void Start()
    {
        move_action = InputSystem.actions.FindAction("Player/Move");
        rotate_action = InputSystem.actions.FindAction("Player/Rotate");

        UpdatePositionnementText();
    }

    void FixedUpdate()
    {
        HandleControls();        
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

            // A full scan has just been produced: lidar.measurements is ready.

            // A "fail" is any scan the estimator could not place reliably: either
            // it rejected the scan outright, or the residual left over is larger
            // than the tolerance used to warn below.
            positionnement_total++;
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

            float residual_x = estimated_position.pos_x - real_position.pos_x;
            float residual_y = estimated_position.pos_y - real_position.pos_y;
            float residual_a = estimated_position.pos_a - real_position.pos_a;

            float residual_distance = Mathf.Sqrt(residual_x * residual_x + residual_y * residual_y);

            residual_x_mm = residual_x;
            residual_y_mm = residual_y;
            residual_a_deg = residual_a;
            UpdatePositionnementText();

            // --- Diagnostic record -------------------------------------------
            // The score at the ground-truth pose, against the score at the pose
            // the search returned. A positive gap means the truth scored better,
            // so the search stopped short of a peak it could have reached and the
            // schedule is at fault; a non-positive gap means the returned pose was
            // the best available and the objective itself is biased, which no
            // search change can fix.
            float score_at_truth = PosEstimator.ScoreAt(lidar.measurements, real_position, lidar_offset);
            float score_gap = score_at_truth - PosEstimator.LastBestScore;

            // Kept for inspection in the debugger: the failure counters and the
            // score comparison are the evidence trail if the residual regresses.
            _ = score_gap;

            if (PosEstimator.LastEstimateWasRejected || residual_distance > 5f || Mathf.Abs(residual_a) > 0.5f)
            {
                positionnement_fails++;
            }

            lidar.BeginSweep();
        }
    }

    /// <summary>
    /// Shows how many positionnement attempts failed out of the total attempted,
    /// the ground-truth residual of the latest estimate, and how long that
    /// estimate took.
    ///
    /// The timing is read from PosEstimator's diagnostics rather than measured
    /// here, so it covers the estimate itself and excludes the surrounding frame
    /// work. It is the same figure the warning log reports.
    /// </summary>
    private void UpdatePositionnementText()
    {
        if (positionnement_text == null)
        {
            return;
        }

        positionnement_text.text =
            $"Fails: {positionnement_fails}/{positionnement_total}\n" +
            $"Residual: ({residual_x_mm:F1}, {residual_y_mm:F1}) mm, {residual_a_deg:F2} deg\n" +
            $"Time: {PosEstimator.LastEstimateMs:F1} ms " +
            $"({PosEstimator.LastSweepCount} sweeps)";
    }

    /// <summary>
    /// Moves the robot with WASD (Player/Move) and rotates it with Q/E (Player/Rotate).
    /// </summary>
    private void HandleControls()
    {
        Vector2 move = move_action != null ? move_action.ReadValue<Vector2>() : Vector2.zero;

        Vector3 translation = new Vector3(-move.y, 0f, move.x) * (movement_speed * Time.deltaTime);
        transform.Translate(translation, Space.World);

        float turn = rotate_action != null ? rotate_action.ReadValue<float>() : 0f;

        transform.Rotate(0f, turn * rotation_speed * Time.deltaTime, 0f, Space.World);
    }

    /// <summary>
    /// Refines the robot's position using the latest lidar scan.
    /// </summary>
    public Pos EstimatePosition(Pos approximate_position, List<Lidar.Measurement> measurements, Pos lidar_offset)
    {
        return PosEstimator.EstimatePosition(approximate_position, measurements, lidar_offset);
    }
}
