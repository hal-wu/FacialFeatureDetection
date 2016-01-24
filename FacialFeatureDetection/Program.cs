using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;

namespace FacialFeatureDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            PupilDetector pupilDetector = new PupilDetector("C:\\Users\\Phoenix\\Documents\\Visual Studio 2013\\Projects\\testPupilDetection\\res\\haarcascade_frontalface_alt.xml");

            Mat frame = new Mat();
            Mat frameCopy = new Mat();
            Capture capture = new Capture(0);
            if (capture == null)
                return;

            while (true)
            {
                frame = capture.QueryFrame();
                frame.CopyTo(frameCopy);
                CvInvoke.Flip(frameCopy, frameCopy, FlipType.Horizontal);
                if (!frameCopy.IsEmpty)
                {
                    Rectangle face = new Rectangle(new Point(0, 0), new Size(0, 0));
                    Point leftPupil = new Point(0, 0);
                    Point rightPupil = new Point(0, 0);
                    pupilDetector.Detect(frameCopy, ref face, ref leftPupil, ref rightPupil);
                    CvInvoke.Rectangle(frameCopy, face, new MCvScalar(1234));
                    CvInvoke.Circle(frameCopy, leftPupil, 3, new MCvScalar(1234));
                    CvInvoke.Circle(frameCopy, rightPupil, 3, new MCvScalar(1234));
                    CvInvoke.Imshow("Pupil Detection", frameCopy);
                }
                else
                {
                    Console.WriteLine(" --(!) No captured frame -- Break!");
                    break;
                }

                int c = CvInvoke.WaitKey(10);
                if ((char)c == 'c')
                    break;
            }
        }
    }
}
