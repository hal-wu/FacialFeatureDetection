using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Drawing;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;

namespace FacialFeatureDetection
{
    public class PupilDetector
    {
        #region Fields

        // Rough eye region percentage in the whole face region

        double _eyeTopPercent = 0.25;

        double _eyeSidePercent = 0.13;

        double _eyeHeightPercent = 0.30;

        double _eyeWidthPercent = 0.35;

        // Preprocessing

        bool _isFaceSmoothNeeded = true;

        double _faceSmoothFactor = 0.005;

        // Algorithm Parameters

        int _fastEyeWidth = 30;

        int _weightBlurSize = 5;

        bool _isWeightEnabled = true;

        double _gradientThreshold = 0.3;

        // Post processing

        bool _isPostProcessingEnabled = true;

        double _postProcessingThreshold = 0.90;

        CascadeClassifier _faceCascadeClassifier;

        #endregion // Fields

        #region Constructor

        public PupilDetector(string faceCascadePath)
        {
            _faceCascadeClassifier = new CascadeClassifier(faceCascadePath);
        }

        #endregion // Constructor

        #region Properties

        #endregion // Properties

        #region Methods

        private void GetNormalizedImageGradients(
            Matrix<byte> grayImage, 
            out Matrix<double> XOfGradients, 
            out Matrix<double> YOfGradients
            )
        {
            XOfGradients = new Matrix<double>(grayImage.Size);
            YOfGradients = new Matrix<double>(grayImage.Size);
            Matrix<double> magnitudes = new Matrix<double>(grayImage.Size);

            for (int iRow = 0; iRow < grayImage.Rows; iRow++) {
                XOfGradients.Data[iRow, 0] = grayImage.Data[iRow, 1] - grayImage.Data[iRow, 0];
                for (int iCol = 1; iCol < grayImage.Cols - 1; iCol++)
                    XOfGradients.Data[iRow, iCol] = (grayImage.Data[iRow, iCol + 1] - grayImage.Data[iRow, iCol - 1]) / 2;
                XOfGradients.Data[iRow, grayImage.Cols - 1] = grayImage.Data[iRow, grayImage.Cols - 1] - grayImage.Data[iRow, grayImage.Cols - 2];
            }

            for (int iCol = 0; iCol < grayImage.Cols; iCol++) {
                YOfGradients.Data[0, iCol] = grayImage.Data[1, iCol] - grayImage.Data[0, iCol];
                for (int iRow = 1; iRow < grayImage.Rows - 1; iRow++)
                    YOfGradients.Data[iRow, iCol] = (grayImage.Data[iRow + 1, iCol] - grayImage.Data[iRow - 1, iCol]) / 2;
                YOfGradients.Data[grayImage.Rows - 1, iCol] = grayImage.Data[grayImage.Rows - 1, iCol] - grayImage.Data[grayImage.Rows - 2, iCol];
            }

            for (int iRow = 0; iRow < grayImage.Rows; iRow++) {
                for (int iCol = 0; iCol < grayImage.Cols; iCol++) {
                    double xVal = XOfGradients.Data[iRow, iCol];
                    double yVal = YOfGradients.Data[iRow, iCol];
                    magnitudes.Data[iRow, iCol] = Math.Sqrt(xVal * xVal + yVal * yVal);
                }
            }

            MCvScalar stdDev = new MCvScalar();
            MCvScalar mean = new MCvScalar();
            CvInvoke.MeanStdDev(magnitudes, ref mean, ref stdDev);
            double dynamicThreshold = _gradientThreshold * stdDev.V0 + mean.V0;
            for (int iRow = 0; iRow < grayImage.Rows; iRow++) {
                for (int iCol = 0; iCol < grayImage.Cols; iCol++) {
                    double xVal = XOfGradients.Data[iRow, iCol];
                    double yVal = YOfGradients.Data[iRow, iCol];
                    double magnitude = magnitudes.Data[iRow, iCol];
                    if (magnitude > dynamicThreshold) {
                        XOfGradients.Data[iRow, iCol] = xVal / magnitude;
                        YOfGradients.Data[iRow, iCol] = yVal / magnitude;
                    }
                    else {
                        XOfGradients.Data[iRow, iCol] = 0.0;
                        YOfGradients.Data[iRow, iCol] = 0.0;
                    }
                }
            }
        }

        private Matrix<byte> GetSmoothedInvertedImage(Matrix<byte> grayImage)
        {
            Matrix<byte> newImage = new Matrix<byte>(grayImage.Size);
            CvInvoke.GaussianBlur(grayImage, newImage, new Size(_weightBlurSize, _weightBlurSize), 0, 0);
            CvInvoke.BitwiseNot(newImage, newImage);
            return newImage;
        }

        private Matrix<double> GetObjectiveFunction(
            ref Matrix<double> XOfGradients,
            ref Matrix<double> YOfGradients,
            ref Matrix<byte> weight
            )
        {
            Matrix<double> result = new Matrix<double>(weight.Size);
            result.SetValue(0);
            double numGradients = weight.Rows * weight.Cols;
            for (int iRow = 0; iRow < weight.Rows; iRow++) {
                for (int iCol = 0; iCol < weight.Cols; iCol++) {
                    double gx = XOfGradients.Data[iRow, iCol];
                    double gy = YOfGradients.Data[iRow, iCol];
                    if (gx == 0.0 && gy == 0.0)
                        continue;
                    for (int cRow = 0; cRow < weight.Rows; cRow++) {
                        for (int cCol = 0; cCol < weight.Cols; cCol++) {
                            if (iCol == cCol && iRow == cRow)
                                continue;
                            double dx = iCol - cCol;
                            double dy = iRow - cRow;
                            double mag = Math.Sqrt(dx * dx + dy * dy);
                            dx = dx / mag;
                            dy = dy / mag;
                            double dotProduct = Math.Max(0.0, dx * gx + dy * gy);
                            result[cRow, cCol] += dotProduct * dotProduct;
                        }
                    }
                }
            }

            for (int cRow = 0; cRow < weight.Rows; cRow++) {
                for (int cCol = 0; cCol < weight.Cols; cCol++) {
                    result[cRow, cCol] *= 1.0 / numGradients;
                    if (_isWeightEnabled)
                        result[cRow, cCol] *= weight.Data[cRow, cCol];
                }
            }

            return result;
        }

        private Point FindEyeCentreHelper(Mat matEyeROI)
        {
            //Scale
            int fastEyeHeight = Convert.ToInt32(Convert.ToDouble(_fastEyeWidth) / matEyeROI.Cols * matEyeROI.Rows);
            Size fastSize = new Size(_fastEyeWidth, fastEyeHeight);
            Matrix<Byte> eyeROI = new Matrix<Byte>(fastSize);
            CvInvoke.Resize(
                matEyeROI,
                eyeROI,
                fastSize
                );

            //Compute normalized image gradients
            Matrix<double> gradientX;
            Matrix<double> gradientY;
            GetNormalizedImageGradients(eyeROI, out gradientX, out gradientY);

            //Compute smoothed and inverted image as weight
            Matrix<byte> weight = GetSmoothedInvertedImage(eyeROI);

            //Compute objective function
            Matrix<double> objectiveFunction = GetObjectiveFunction(ref gradientX, ref gradientY, ref weight);

            //Search maximum of the objective function
            Point maxP = new Point();
            Point minP = new Point();
            double maxVal = 0;
            double minVal = 0;
            CvInvoke.MinMaxLoc(objectiveFunction, ref minVal, ref maxVal, ref minP, ref maxP);
            if (_isPostProcessingEnabled) {
                DoPostProcessing(maxVal, ref objectiveFunction);
                CvInvoke.MinMaxLoc(objectiveFunction, ref minVal, ref maxVal, ref minP, ref maxP);
            }

            //Unscale
            double ratio = (Convert.ToDouble(_fastEyeWidth) / matEyeROI.Width);
            int x = Convert.ToInt32(Math.Round(maxP.X / ratio));
            int y = Convert.ToInt32(Math.Round(maxP.Y / ratio));
            return new Point(x, y);
        }

        private bool IsPointInMatrix(Point pt, Matrix<double> mat)
        {
            return pt.X >= 0 && pt.X < mat.Cols && pt.Y >= 0 && pt.Y < mat.Rows;
        }

        private void DoPostProcessing(double maxOfObjectiveFunc, ref Matrix<double> objectiveFunc)
        {
            double threshold = maxOfObjectiveFunc * _postProcessingThreshold;
            for (int iRow = 0; iRow < objectiveFunc.Rows; iRow++) {
                for (int iCol = 0; iCol < objectiveFunc.Cols; iCol++) {
                    if (objectiveFunc[iRow, iCol] < threshold)
                        objectiveFunc[iRow, iCol] = 0.0;
                }
            }

            CvInvoke.Rectangle(
                objectiveFunc, 
                new Rectangle(0, 0, objectiveFunc.Cols, objectiveFunc.Rows), 
                new MCvScalar(255.0)
                );
            Queue<Point> toDo = new Queue<Point>();
            toDo.Enqueue(new Point(0, 0));
            while (toDo.Count > 0)
            {
                Point p = toDo.Dequeue();
                if (objectiveFunc.Data[p.Y, p.X] == 0.0)
                    continue;

                // Left
                Point npl = new Point(p.X - 1, p.Y);
                if (IsPointInMatrix(npl, objectiveFunc)) toDo.Enqueue(npl);

                // Right
                Point npr = new Point(p.X + 1, p.Y);
                if (IsPointInMatrix(npr, objectiveFunc)) toDo.Enqueue(npr);

                // Up
                Point npu = new Point(p.X, p.Y - 1);
                if (IsPointInMatrix(npu, objectiveFunc)) toDo.Enqueue(npu);

                // Down
                Point npd = new Point(p.X, p.Y + 1); 
                if (IsPointInMatrix(npd, objectiveFunc)) toDo.Enqueue(npd);

                // To zero
                objectiveFunc[p.Y, p.X] = 0.0;
            }
        }

        private void FindEyeCentres(
            Mat grayImage,
            Rectangle face,
            out Point leftEyeCentre,
            out Point rightEyeCentre)
        {
            Mat faceROI = new Mat(grayImage, face);

            if (_isFaceSmoothNeeded)
            {
                double sigma = _faceSmoothFactor * face.Width;
                CvInvoke.GaussianBlur(faceROI, faceROI, new Size(0, 0), sigma);
            }

            // Find eye regions
            int eyeRegionwidth = Convert.ToInt32(face.Width * _eyeWidthPercent);
            int eyeRegionHeight = Convert.ToInt32(face.Width * _eyeHeightPercent);
            int eyeRegiontop = Convert.ToInt32(face.Height * _eyeTopPercent);
            Rectangle leftEyeRegion = new Rectangle(
                Convert.ToInt32(face.Width * _eyeSidePercent),
                eyeRegiontop,
                eyeRegionwidth,
                eyeRegionHeight
                );
            Rectangle rightEyeRegion = new Rectangle(
                Convert.ToInt32(face.Width - eyeRegionwidth - face.Width * _eyeSidePercent),
                eyeRegiontop,
                eyeRegionwidth,
                eyeRegionHeight
                );

            // Find Eye Centres
            leftEyeCentre = FindEyeCentreHelper(new Mat(faceROI, leftEyeRegion));
            rightEyeCentre = FindEyeCentreHelper(new Mat(faceROI, rightEyeRegion));

            // Change eye centres to face coordinates
            rightEyeCentre.X += rightEyeRegion.X + face.X;
            rightEyeCentre.Y += rightEyeRegion.Y + face.Y;
            leftEyeCentre.X += leftEyeRegion.X + face.X;
            leftEyeCentre.Y += leftEyeRegion.Y + face.Y;
        }

        public void Detect(
            Mat image,
            ref Rectangle face,
            ref Point leftEyeCentre,
            ref Point rightEyeCentre
            )
        {
            Mat grayImage = new Mat();

            CvInvoke.CvtColor(image, grayImage, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);

            //normalizes brightness and increases contrast of the image
            CvInvoke.EqualizeHist(grayImage, grayImage);

            // Detect faces
            Rectangle[] faces = _faceCascadeClassifier.DetectMultiScale(
                grayImage,
                1.1,
                10,
                new Size(20, 20)
                );

            if (faces.Length > 0)
            {
                face = faces[0];
                FindEyeCentres(grayImage, faces[0], out leftEyeCentre, out rightEyeCentre);
            }
        }

        #endregion // Methods
    }
}
