using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(Camera))]
public class OnboardCamScript : MonoBehaviour
{
    [Tooltip("Key that triggers the screenshot.")]
    public Key captureKey = Key.Space;

    [Tooltip("Folder (relative to the project) where the images are saved.")]
    public string saveFolder = "Captures";

    void Update()
    {
        Keyboard keyboard = Keyboard.current;

        if (keyboard != null && keyboard[captureKey].wasPressedThisFrame)
        {
            Capture();
        }
    }

    void Capture()
    {
        RenderTexture rt = GetComponent<Camera>().targetTexture;
        if (rt == null)
        {
            Debug.LogWarning("OnboardCamScript: camera has no target RenderTexture.");
            return;
        }

        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = rt;
        Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        tex.Apply();
        RenderTexture.active = previous;

        // Pose in the estimator's frame: mm, X forward, Y left, yaw CCW.
        // Unity Z maps to estimator Y with a sign flip, and yaw is negated
        // because Unity turns clockwise about +Y.
        Vector3 p = transform.position * 1000f;
        float x = p.x;
        float y = p.z;
        float yaw = -transform.eulerAngles.y+90;
        if (yaw > 180) yaw -= 360;
        float pitch = transform.eulerAngles.x;

        // Invariant culture keeps the decimal point a point, so the name parses
        // the same on every machine.
        string pose =
            $"x{x.ToString("F1", CultureInfo.InvariantCulture)}" +
            $"_y{y.ToString("F1", CultureInfo.InvariantCulture)}" +
            $"_yaw{yaw.ToString("F1", CultureInfo.InvariantCulture)}" +
            $"_pitch{pitch.ToString("F1", CultureInfo.InvariantCulture)}";

        Directory.CreateDirectory(saveFolder);
        string path = Path.Combine(
            saveFolder,
            $"capture_{System.DateTime.Now:yyyyMMdd_HHmmss}_{pose}.png"
        );

        File.WriteAllBytes(path, tex.EncodeToPNG());
        Destroy(tex);

        Debug.Log($"Screenshot saved to {Path.GetFullPath(path)}");
    }
}
