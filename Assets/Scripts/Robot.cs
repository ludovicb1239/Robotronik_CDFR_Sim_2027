using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
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

    /// <summary>
    /// One diagnostic record per scan, kept in memory so the whole run can be
    /// dumped to CSV when the scene stops. Written whether or not the estimate
    /// passed, so the passing scans are available as a baseline to compare the
    /// failing ones against.
    /// </summary>
    private readonly List<ScanRecord> scan_records = new List<ScanRecord>();

    /// <summary>
    /// Everything needed to explain one estimate after the fact.
    ///
    /// The fields fall into three groups. The residual and prior error say what
    /// happened. The three scores - at the prior, at the returned pose, and at
    /// the ground truth - say whether the search stopped early or the objective
    /// is biased. The sweep telemetry says which part of the schedule misbehaved.
    /// </summary>
    private struct ScanRecord
    {
        public float prior_x;
        public float prior_y;
        public float prior_a;

        public float residual_x;
        public float residual_y;
        public float residual_a;

        public bool rejected;

        public float base_score;
        public float best_score;
        public float score_at_truth;
        public float score_gap;

        public int clamped_sweeps;
        public int improving_sweeps;

        public float final_range_mm;
        public float final_angle_range_deg;

        public double ms;
        public int ray_count;
    }

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
            // One line per scan, written whether or not the estimate passed, so
            // the log can be analysed offline without re-running the scene.
            //
            // The decisive number is score_gap: the objective's score at the
            // ground-truth pose minus its score at the pose we returned.
            //
            //   score_gap > 0  the truth scores better, so the search stopped
            //                  short of a peak it could have reached. The
            //                  schedule is at fault - most likely the zoom
            //                  shrank the range past the remaining correction.
            //   score_gap <= 0 the returned pose scores at least as well as the
            //                  truth, so the search did find the best peak and
            //                  the objective itself is biased. No search change
            //                  can fix that; the score has to change.
            //
            // clamped_sweeps counts sweeps that settled on the outermost sample
            // offered, which is the direct symptom of the range being too small
            // for the remaining travel. improving_sweeps counts sweeps that
            // raised the score at all; when it is far below the sweep count, the
            // later sweeps were flat, which points at the objective instead.
            float score_at_truth = PosEstimator.ScoreAt(lidar.measurements, real_position, lidar_offset);
            float score_gap = score_at_truth - PosEstimator.LastBestScore;

            ScanRecord record = new ScanRecord
            {
                residual_x = residual_x,
                residual_y = residual_y,
                residual_a = residual_a,
                prior_x = approximate_position.pos_x - real_position.pos_x,
                prior_y = approximate_position.pos_y - real_position.pos_y,
                prior_a = approximate_position.pos_a - real_position.pos_a,
                rejected = PosEstimator.LastEstimateWasRejected,
                base_score = PosEstimator.LastBaseScore,
                best_score = PosEstimator.LastBestScore,
                score_at_truth = score_at_truth,
                score_gap = score_gap,
                clamped_sweeps = PosEstimator.LastClampedSweeps,
                improving_sweeps = PosEstimator.LastImprovingSweeps,
                final_range_mm = PosEstimator.LastFinalRangeMm,
                final_angle_range_deg = PosEstimator.LastFinalAngleRangeDeg,
                ms = PosEstimator.LastEstimateMs,
                ray_count = lidar.measurements.Count,
            };

            scan_records.Add(record);

            if (PosEstimator.LastEstimateWasRejected || residual_distance > 5f || Mathf.Abs(residual_a) > 0.5f)
            {
                positionnement_fails++;

                Debug.LogWarning($"Could not find exact position: error " +
                                 $"({approximate_position.pos_x - real_position.pos_x:F0}, " +
                                 $"{approximate_position.pos_y - real_position.pos_y:F0}, " +
                                 $"{approximate_position.pos_a - real_position.pos_a:F1}) " +
                                 $"-> residual " +
                                 $"({residual_x:F0}, {residual_y:F0}, {residual_a:F1}) | " +
                                 $"{PosEstimator.LastEstimateMs:F1} ms, " +
                                 $"{PosEstimator.LastSweepCount} sweeps | " +
                                 $"score base {PosEstimator.LastBaseScore:F3} best {PosEstimator.LastBestScore:F3} " +
                                 $"truth {score_at_truth:F3} gap {score_gap:+0.000;-0.000;0.000} | " +
                                 $"clamped {PosEstimator.LastClampedSweeps} " +
                                 $"improving {PosEstimator.LastImprovingSweeps} | " +
                                 $"final range {PosEstimator.LastFinalRangeMm:F1} mm " +
                                 $"{PosEstimator.LastFinalAngleRangeDeg:F2} deg | " +
                                 $"{PosEstimator.LastWallHitCount}/{lidar.measurements.Count} wall hits");
            }

            lidar.BeginSweep();
        }
    }

    /// <summary>
    /// Writes every recorded scan to a CSV next to the project's persistent data,
    /// so a run can be analysed offline instead of by reading the console. Called
    /// on quit; the file is overwritten each run.
    ///
    /// The header row names each column, and the score columns are what the
    /// analysis keys on. A convenient way to read it back is with pandas, which
    /// handles the columns directly.
    /// </summary>
    private void OnApplicationQuit()
    {
        if (scan_records.Count == 0)
        {
            return;
        }

        // UTF8 without a byte order mark, so the header line is not prefixed with
        // an invisible BOM that would corrupt the first column's name.
        UTF8Encoding encoding = new UTF8Encoding(false);

        string path = Path.Combine(Application.persistentDataPath, "scan_records.csv");

        StringBuilder builder = new StringBuilder();

        builder.AppendLine("prior_x,prior_y,prior_a,residual_x,residual_y,residual_a," +
                           "rejected,base_score,best_score,score_at_truth,score_gap," +
                           "clamped_sweeps,improving_sweeps,final_range_mm," +
                           "final_angle_range_deg,ms,ray_count");

        // Invariant culture throughout: a locale that writes decimal commas would
        // otherwise produce a file that no longer parses as CSV.
        CultureInfo invariant = CultureInfo.InvariantCulture;

        for (int i = 0; i < scan_records.Count; i++)
        {
            ScanRecord r = scan_records[i];

            builder.Append(r.prior_x.ToString("F1", invariant)).Append(',');
            builder.Append(r.prior_y.ToString("F1", invariant)).Append(',');
            builder.Append(r.prior_a.ToString("F3", invariant)).Append(',');
            builder.Append(r.residual_x.ToString("F1", invariant)).Append(',');
            builder.Append(r.residual_y.ToString("F1", invariant)).Append(',');
            builder.Append(r.residual_a.ToString("F3", invariant)).Append(',');
            builder.Append(r.rejected ? '1' : '0').Append(',');
            builder.Append(r.base_score.ToString("F4", invariant)).Append(',');
            builder.Append(r.best_score.ToString("F4", invariant)).Append(',');
            builder.Append(r.score_at_truth.ToString("F4", invariant)).Append(',');
            builder.Append(r.score_gap.ToString("F4", invariant)).Append(',');
            builder.Append(r.clamped_sweeps).Append(',');
            builder.Append(r.improving_sweeps).Append(',');
            builder.Append(r.final_range_mm.ToString("F2", invariant)).Append(',');
            builder.Append(r.final_angle_range_deg.ToString("F3", invariant)).Append(',');
            builder.Append(r.ms.ToString("F1", invariant)).Append(',');
            builder.Append(r.ray_count);
            builder.AppendLine();
        }

        File.WriteAllText(path, builder.ToString(), encoding);

        Debug.Log($"Wrote {scan_records.Count} scan records to {path}");
    }

    /// <summary>
    /// Shows how many positionnement attempts failed out of the total attempted,
    /// the ground-truth residual of the latest estimate, and how long that
    /// estimate took.
    ///
    /// The timing is read from PosEstimator's diagnostics rather than measured
    /// here, so it covers the estimate itself and excludes the surrounding frame
    /// work. It is the same figure the warning log and the CSV record.
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
