using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(Camera))]
public class OnboardCamScript : MonoBehaviour
{
    [Tooltip("Folder (relative to the project) where the images are saved.")]
    public string saveFolder = "Captures";

    private int i = 0;

    // Pose the camera had when the frame now in the target texture was drawn.
    private Vector3 renderedPosition;
    private Vector3 renderedEuler;
    private bool hasRenderedFrame;

    private void OnEnable()
    {
        // URP does not raise Camera.onPreCull - that callback only exists in the
        // built-in pipeline - so subscribing to it here would leave the capture
        // silent. RenderPipelineManager.beginCameraRendering is the equivalent
        // hook, and it fires immediately before a camera is rendered, with that
        // camera's pose for the frame already locked in.
        RenderPipelineManager.beginCameraRendering += HandleBeginCameraRendering;
    }

    private void OnDisable()
    {
        RenderPipelineManager.beginCameraRendering -= HandleBeginCameraRendering;
    }

    private void HandleBeginCameraRendering(ScriptableRenderContext context, Camera camera)
    {
        Camera self = GetComponent<Camera>();

        // This callback is static, so it is offered every camera in the scene.
        // Only this one should drive captures.
        if (camera != self || !self.isActiveAndEnabled)
        {
            return;
        }

        Debug.Log($"OnboardCamScript: fired for '{camera.name}' on frame {Time.frameCount}.");

        // The target texture still holds the previous frame, so save it against
        // the pose recorded for it, then keep this frame's pose for the next one.
        if (hasRenderedFrame)
        {
            Capture(renderedPosition, renderedEuler);
        }

        renderedPosition = transform.position;
        renderedEuler = transform.eulerAngles;
        hasRenderedFrame = true;
    }

    private void Capture(Vector3 cameraPosition, Vector3 cameraEuler)
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
        Vector3 p = cameraPosition * 1000f;
        float x = p.x;
        float y = p.z;
        float yaw = -cameraEuler.y+90;
        if (yaw > 180) yaw -= 360;
        float pitch = cameraEuler.x;

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
            $"capture_{i}_{pose}.png"
        );
        i++;

        File.WriteAllBytes(path, tex.EncodeToPNG());
        Destroy(tex);

        Debug.Log($"Screenshot saved to {Path.GetFullPath(path)}");
    }
}
