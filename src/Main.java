import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.videoio.VideoCapture;

public class Main {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {

        VideoCapture camera = new VideoCapture(0);

        if (!camera.isOpened()) {
            System.out.println("Camera not found");
            return;
        }

        Mat frame = new Mat();

        while (true) {

            camera.read(frame);

            HighGui.imshow("Gesture Camera", frame);

            if (HighGui.waitKey(1) == 27) {
                break;
            }
        }

        camera.release();
    }
}