
#include <QtCore/QCoreApplication>
#include <QFile>
#include <QTextStream>
// includes of STD libs
#include <iostream>
#include <string>
#include <time.h>

// includes of OpenCV library
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video\tracking.hpp>

#include "Particle.h"

#define EPSILON 0.0001

using namespace std;
using namespace cv;

// GLOBAL VARIABLES FOR CALLBACKS //

int g_initParticleLeftX, g_initParticleLeftY, g_initParticleRightX, g_initParticleRightY;
bool g_leftCameraInitialized = false, g_rightCameraInitialized = false;

void onMouseLeftCamera(int event, int x, int y, int flags, void* param)
{
	if(event != CV_EVENT_LBUTTONDOWN)
		return;
	if(g_leftCameraInitialized == false)
	{
		g_initParticleLeftX = x;
		g_initParticleLeftY = y;

		cout << "Left camera initialized with coordinates: " << g_initParticleLeftX << "; " << g_initParticleLeftY << endl;

		g_leftCameraInitialized = true;
	}
}

void onMouseRightCamera(int event, int x, int y, int flags, void* param)
{
	if(event != CV_EVENT_LBUTTONDOWN)
		return;

	if(g_rightCameraInitialized == false)
	{
		g_initParticleRightX = x;
		g_initParticleRightY = y;

		cout << "Right camera initialized with coordinates: " << g_initParticleRightX << "; " << g_initParticleRightY << endl;

		g_rightCameraInitialized = true;
	}
}


Mat_<double> LinearLSTriangulation(Point3d u, Matx34d P, Point3d u1, Matx34d P1)
{
    Matx43d A(u.x*P(2,0)-P(0,0),    u.x*P(2,1)-P(0,1),      u.x*P(2,2)-P(0,2),
          u.y*P(2,0)-P(1,0),    u.y*P(2,1)-P(1,1),      u.y*P(2,2)-P(1,2),
          u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),   u1.x*P1(2,2)-P1(0,2),
          u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),   u1.y*P1(2,2)-P1(1,2));
   
	Mat_<double> B = (Mat_<double>(4,1) <<    -(u.x*P(2,3)    -P(0,3)),
                      -(u.y*P(2,3)  -P(1,3)),
                      -(u1.x*P1(2,3)    -P1(0,3)),
                      -(u1.y*P1(2,3)    -P1(1,3)));
 
    Mat_<double> X(3,1);
    solve(A,B,X,DECOMP_SVD);
    return X;
}

Point3d IterativeLinearLSTriangulation(Point3d u, Matx34d P, Point3d u1, Matx34d P1) {
    double wi = 1, wi1 = 1;
    Mat_<double> X(4,1); 
	Mat_<double> X_ = LinearLSTriangulation(u,P,u1,P1);
	X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
    for (int i=0; i<10; i++) {
        double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
        double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);

		if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON)
		{
			//cout << "Epsilon achieved." << endl;
			break;
		}
         
        wi = p2x;
        wi1 = p2x1;
         
        Matx43d A((u.x*P(2,0)-P(0,0))/wi,       (u.x*P(2,1)-P(0,1))/wi,         (u.x*P(2,2)-P(0,2))/wi,     
                  (u.y*P(2,0)-P(1,0))/wi,       (u.y*P(2,1)-P(1,1))/wi,         (u.y*P(2,2)-P(1,2))/wi,     
                  (u1.x*P1(2,0)-P1(0,0))/wi1,   (u1.x*P1(2,1)-P1(0,1))/wi1,     (u1.x*P1(2,2)-P1(0,2))/wi1, 
                  (u1.y*P1(2,0)-P1(1,0))/wi1,   (u1.y*P1(2,1)-P1(1,1))/wi1,     (u1.y*P1(2,2)-P1(1,2))/wi1);
        
		Mat_<double> B = (Mat_<double>(4,1) <<    -(u.x*P(2,3)    -P(0,3))/wi,
                          -(u.y*P(2,3)  -P(1,3))/wi,
                          -(u1.x*P1(2,3)    -P1(0,3))/wi1,
                          -(u1.y*P1(2,3)    -P1(1,3))/wi1);
         
        solve(A,B,X_,DECOMP_SVD);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
    }
    return Point3d(X(0),X(1),X(2));
}

vector<Particle> initialRandomParticles(int numberOfParticles, int rangeForRandomParticle, int imageWidth, int imageHeight, Point initialPosition)
{
	vector<Particle> particles;
	for(int i=0; i < numberOfParticles; i++)
	{
		int randX = rand() % rangeForRandomParticle;
		randX -= rangeForRandomParticle/2;
		int randY = rand() % rangeForRandomParticle;
		randY -= rangeForRandomParticle/2;
		int particleX = initialPosition.x + randX;
		int particleY = initialPosition.y + randY;
		if(particleX < 10)
			particleX = 10;
		if(particleY < 10)
			particleY = 10;
		if(particleX > (imageWidth - 10))
			particleX = imageWidth - 10;
		if(particleY > (imageHeight - 10))
			particleY = imageHeight - 10;
		Particle nextParticle(particleX, particleY, 0, 0);
		particles.push_back(nextParticle);
	}
	return particles;
}

vector<Particle> randomNumberOfParticle(int numberOfParticles, int rangeForRandomParticle, int imageWidth, int imageHeight, int number, int x, int y, vector<int> speed)
{
	int speedX = speed[0];
	int speedY = speed[1];
	vector<Particle> _particles;
	for(int j = 0; j < number; j++)
	{
		int randX = rand() % rangeForRandomParticle;
		randX -= (rangeForRandomParticle/2);
		int randY = rand() % rangeForRandomParticle;
		randY -= (rangeForRandomParticle/2);

		int particleX = x + randX + speedX;
		int particleY = y + randY + speedY;

		if(particleX < 10)
			particleX = 10;
		if(particleY < 10)
			particleY = 10;
		if(particleX > (imageWidth - 10))
			particleX = imageWidth - 10;
		if(particleY > (imageHeight - 10))
			particleY = imageHeight - 10;
		Particle nextParticle(particleX, particleY, 0, 0);
		_particles.push_back(nextParticle);
	}
	return _particles;
}

void drawObject(int x, int y, Mat &frame, char param)
{
	if(param == 'C')
	{
		circle(frame,Point(x,y),20,Scalar(0,255,0),2);

		if(y-25>0)
			line(frame,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
		else 
			line(frame,Point(x,y),Point(x,0),Scalar(0,255,0),2);
		if(y+25<frame.size().height)
			line(frame,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
		else 
			line(frame,Point(x,y),Point(x,frame.size().height),Scalar(0,255,0),2);
		if(x-25>0)
			line(frame,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
		else 
			line(frame,Point(x,y),Point(0,y),Scalar(0,255,0),2);
		if(x+25<frame.size().width)
			line(frame,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);
		else 
			line(frame,Point(x,y),Point(frame.size().width,y),Scalar(0,255,0),2);
	}
	else if(param == 'P')
	{
		circle(frame,Point(x,y),2,Scalar(255,255,255),1);
	}
}

int main()
{
	VideoCapture camera1, camera2;
	Mat image1, image2,grayImage1, grayImage2;
	int particlesNumber = 120;
	int rangeForRandom = 50;
	int particlesToDelete = 90;
	string leftCameraFilename, rightCameraFilename, stereoData;
	Mat cameraMatrix[2], distCoeffs[2];
	Mat translationVector;
	Mat rotationMatrix;
	Mat mapLeft1, mapLeft2, mapRight1, mapRight2;
	Mat particleROILeft, particleROIRight;
	MatND patternHistogramLeft, patternHistogramRight;
	vector<Particle> particlesLeft, particlesRight;
	vector<int> leftSpeed, rightSpeed;
	bool firstSpeedEstimation = true;
	Point estimatedPositionLeft(0,0), estimatedPositionRight(0,0), formerEstimatedPositionLeft(0,0), formerEstimatedPositionRight(0,0);
	KalmanFilter KF(6, 3, 0);
	Mat_<float> state(6,1), measurement(3,1);
	measurement.setTo(Scalar(0));
	Mat processNoise(6, 1, CV_32F);

	// logger file settings
	QFile loggerFileLeft;
	QTextStream textStreamLeft;
	loggerFileLeft.setFileName(QString("outputData/trackedPathLeft.txt"));
	loggerFileLeft.open(QIODevice::ReadWrite);
	textStreamLeft.setDevice(&loggerFileLeft);
	QFile loggerFileRight;
	QTextStream textStreamRight;
	loggerFileRight.setFileName(QString("outputData/trackedPathRight.txt"));
	loggerFileRight.open(QIODevice::ReadWrite);
	textStreamRight.setDevice(&loggerFileRight);
	QTextStream triangulatedTextStream;
	QFile triangulatedFile;
	triangulatedFile.setFileName("triangulatedPoints/results.txt");
	triangulatedFile.open(QIODevice::ReadWrite);
	triangulatedTextStream.setDevice(&triangulatedFile);
	QTextStream triangulatedKalmanTextStream;
	QFile triangulatedKalmanFile;
	triangulatedKalmanFile.setFileName("triangulatedPoints/kalman-results.txt");
	triangulatedKalmanFile.open(QIODevice::ReadWrite);
	triangulatedKalmanTextStream.setDevice(&triangulatedKalmanFile);
	// FOR CHESSBOARD TRIANGULATION ONLY
	QFile loggerFile;
	QTextStream textStream;
	loggerFile.setFileName(QString("outputData/triangulatedPoints-%1.txt").arg(0));
	loggerFile.open(QIODevice::ReadWrite);
	textStream.setDevice(&loggerFile);
	int board_w;
	int board_h;
	Size board_size;

	// histogram settings
	int hbins = 12, sbins = 12;
	int histogramSize[] = {hbins, sbins};
	int channels[] = {0, 1};
	float hranges[] = {0, 180};
	float sranges[] = {0, 255};
	const float * ranges[] = {hranges, sranges};
	// end of histogram settings

	srand(time(NULL));

	// PARTICLE FILTER INITIALIZATION
	//camera1.open("video/Camera1.avi");
	//camera2.open("video/Camera2.avi");
	/*if (!camera1.isOpened())
    {
        cout  << "Could not open the input video camera1" << endl;
    }
	if (!camera2.isOpened())
    {
        cout  << "Could not open the input video camera2" << endl;
    }*/

	/*cout << "Input filename prefix for your left camera calibration files (i.e. left-):" << endl;
	cin >> leftCameraFilename;
	cout << "Filename saved. Your filename for left camera: " << leftCameraFilename << endl;

	cout << "Input filename prefix for your left camera calibration files (i.e. left-):" << endl;
	cin >> rightCameraFilename;
	cout << "Filename saved. Your filename for left camera: " << rightCameraFilename << endl;

	cout << "Input filename for stereo data: (i.e. stereoCalibration)" << endl;
	cin >> stereoData;
	cout << "Filename saved. Your filename for stereoCalibration: " << stereoData << endl;*/
	
	cout << "Input board width: ";
	cin >> board_w;
	cout << endl << "Input board height: ";
	cin >> board_h;
	cout << endl << "Board width: " << board_w << ", board height: " << board_h << endl;
	board_size = Size(board_w, board_h);
	leftCameraFilename = "left-";
	rightCameraFilename = "right-";
	stereoData = "stereoCalibration";

	cv::FileStorage fileIntLeft(QString("calibration/%1intrinsic.xml").arg(leftCameraFilename.data()).toStdString(), cv::FileStorage::READ);
	cv::FileStorage fileIntRight(QString("calibration/%1intrinsic.xml").arg(rightCameraFilename.data()).toStdString(), cv::FileStorage::READ);
	cv::FileStorage fileDistLeft(QString("calibration/%1disortion.xml").arg(leftCameraFilename.data()).toStdString(), cv::FileStorage::READ);
	cv::FileStorage fileDistRight(QString("calibration/%1disortion.xml").arg(rightCameraFilename.data()).toStdString(), cv::FileStorage::READ);
	cv::FileStorage fileStereo(QString("calibration/%1.xml").arg(stereoData.data()).toStdString(), cv::FileStorage::READ);
	fileIntLeft["I"] >> cameraMatrix[0];
	fileIntRight["I"] >> cameraMatrix[1];
	fileDistLeft["D"] >> distCoeffs[0];
	fileDistRight["D"] >> distCoeffs[1];
	fileStereo["R"] >> rotationMatrix;
	fileStereo["T"] >> translationVector;

	Matx33d K1(cameraMatrix[0].at<double>(0,0), cameraMatrix[0].at<double>(0,1), cameraMatrix[0].at<double>(0,2),
			   cameraMatrix[0].at<double>(1,0), cameraMatrix[0].at<double>(1,1), cameraMatrix[0].at<double>(1,2),
			   cameraMatrix[0].at<double>(2,0), cameraMatrix[0].at<double>(2,1), cameraMatrix[0].at<double>(2,2));
	Matx33d K2(cameraMatrix[1].at<double>(0,0), cameraMatrix[1].at<double>(0,1), cameraMatrix[1].at<double>(0,2),
			   cameraMatrix[1].at<double>(1,0), cameraMatrix[1].at<double>(1,1), cameraMatrix[1].at<double>(1,2),
			   cameraMatrix[1].at<double>(2,0), cameraMatrix[1].at<double>(2,1), cameraMatrix[1].at<double>(2,2));
	Matx34d RT1(1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0);
	Matx34d RT2(rotationMatrix.at<double>(0,0), rotationMatrix.at<double>(0,1), rotationMatrix.at<double>(0,2), translationVector.at<double>(0,0),
			    rotationMatrix.at<double>(1,0), rotationMatrix.at<double>(1,1), rotationMatrix.at<double>(1,2), translationVector.at<double>(1,0),
			    rotationMatrix.at<double>(2,0), rotationMatrix.at<double>(2,1), rotationMatrix.at<double>(2,2), translationVector.at<double>(2,0));

	cout << "Matrixes loaded. " << endl;

	// undistort init
	image1 = imread(QString("video/camera1-%1.png").arg(0).toStdString());
	image2 = imread(QString("video/camera2-%1.png").arg(0).toStdString());
	initUndistortRectifyMap(
		cameraMatrix[0],  // computed camera matrix
		distCoeffs[0], // computed distortion matrix
		Mat(), // optional rectification (none)
		Mat(), // camera matrix to generate undistorted
		image1.size(),  // size of undistorted
		CV_32FC1,      // type of output map
		mapLeft1, mapLeft2); 
	initUndistortRectifyMap(
		cameraMatrix[1],  // computed camera matrix
		distCoeffs[1], // computed distortion matrix
		Mat(), // optional rectification (none)
		Mat(), // camera matrix to generate undistorted
		image2.size(),  // size of undistorted
		CV_32FC1,      // type of output map
		mapRight1, mapRight2); 
	// end of undistort init

	// CHESSBOARD TRIANGULATION AND SAVE // 
	image1 = imread("video/left.png");
	image2 = imread("video/right.png");
	vector<vector<Point2f> > imagePoints[2];
	vector<vector<Point3f> > objectPoints;
	imagePoints[0].resize(1);
	imagePoints[1].resize(1);
	vector<Point2f>& corners1 = imagePoints[0][0];
	vector<Point3f> obj1;
	vector<Point2f>& corners2 = imagePoints[1][0];
	vector<Point3f> obj2;

	bool found1 = findChessboardCorners( image1, board_size, corners1, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
	bool found2 = findChessboardCorners( image2, board_size, corners2, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
	if(found1 && found2)
	{
		cvtColor(image1, grayImage1, CV_BGR2GRAY);
		cvtColor(image2, grayImage2, CV_BGR2GRAY);
		cornerSubPix(grayImage1, corners1, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
		cornerSubPix(grayImage2, corners2, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
		drawChessboardCorners(image1, board_size, corners1, found1);
		drawChessboardCorners(image2, board_size, corners2, found2);

		Point3d X(0, 0, 0), X_Former(0, 0, 0);

		for(int i=0; i < corners1.size(); i++)
		{
			Point3d pointLeft(corners1[i].x, corners1[i].y, 1.0);
			Point3d pointRight(corners2[i].x, corners2[i].y, 1.0);
			X = IterativeLinearLSTriangulation(pointLeft, K1*RT1, pointRight, K2*RT2);
			textStream << X.x << " " << X.y << " " << X.z << endl;
		}
		cout << "Image ended. " << endl << endl;
		loggerFile.close();
	}
	else
	{
		cout << "No chessboard found." << endl;
	}

	namedWindow("Particle filter - CAM1", 1);
	setMouseCallback("Particle filter - CAM1", onMouseLeftCamera, 0);
	image1 = imread(QString("video/camera1-%1.png").arg(0).toStdString());
	remap(image1, image1, mapLeft1, mapLeft2, INTER_LINEAR);
	cvtColor(image1, image1, CV_RGB2HSV);
	imshow("Particle filter - CAM1", image1);
	while(!g_leftCameraInitialized)
	{	
		waitKey(30);
	}

	namedWindow("Particle filter - CAM2", 1);
	setMouseCallback("Particle filter - CAM2", onMouseRightCamera, 0);
	image2 = imread(QString("video/camera2-%1.png").arg(0).toStdString());
	remap(image2, image2, mapRight1, mapRight2, INTER_LINEAR);
	cvtColor(image2, image2, CV_RGB2HSV);
	imshow("Particle filter - CAM2", image2);
	while(!g_rightCameraInitialized)
	{	
		waitKey(30);
	}

	// CALCULATING INITIAL POSITION FOR KALMAN FILTER
	Point3d result = IterativeLinearLSTriangulation(Point3d(g_initParticleLeftX, g_initParticleLeftY, 1.0), K1*RT1, Point3d(g_initParticleRightX, g_initParticleRightY, 1.0), K2*RT2);
	KF.statePre.at<float>(0) = result.x;
	KF.statePre.at<float>(1) = result.y;
	KF.statePre.at<float>(2) = result.z;
	KF.statePre.at<float>(3) = 0;
	KF.statePre.at<float>(4) = 0;
	KF.statePre.at<float>(5) = 0;
	cout << "Object position initalized." << endl;

	// PATTERN HISTOGRAM CALCULATION
	particleROILeft = Mat(image1, Rect(g_initParticleLeftX - 4, g_initParticleLeftY - 4, 8, 8));
	particleROIRight = Mat(image2, Rect(g_initParticleRightX - 4, g_initParticleRightY - 4, 8, 8));
	calcHist(&particleROILeft, 1, channels, Mat(), patternHistogramLeft, 2, histogramSize, ranges, true, false);
	calcHist(&particleROIRight, 1, channels, Mat(), patternHistogramRight, 2, histogramSize, ranges, true, false);
	cout << "Pattern histogram calculated." << endl;

	// FIRST STEP OF PARTICLE FILTER ALGORITHM - INITIAL RANDOM PARTICLES AND 
	particlesLeft = initialRandomParticles(particlesNumber, rangeForRandom, image1.size().width, image1.size().height, Point(g_initParticleLeftX, g_initParticleLeftY));
	particlesRight = initialRandomParticles(particlesNumber, rangeForRandom, image2.size().width, image2.size().height, Point(g_initParticleRightX, g_initParticleRightY));

	cout << "Entering particle filter loop. " << endl;

	// FINISH KALMAN INITIALIZATION
	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(0.1));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1));
    setIdentity(KF.errorCovPost, Scalar::all(0.1));

	int k = 0;

	while(1)
	{
		// PREPARE IMAGES
		image1 = imread(QString("video/camera1-%1.png").arg(k).toStdString());
		image2 = imread(QString("video/camera2-%1.png").arg(k).toStdString());
		k++;
		if(image1.empty() || image2.empty())
			break;
		remap(image1, image1, mapLeft1, mapLeft2, INTER_LINEAR);
		remap(image2, image2, mapRight1, mapRight2, INTER_LINEAR);
		cvtColor(image1, image1, CV_RGB2HSV);
		cvtColor(image2, image2, CV_RGB2HSV);

		// DRAW PARTICLES
		for(int i = 0; i < particlesLeft.size(); i++)
		{
			drawObject(particlesLeft[i].x, particlesLeft[i].y, image1, 'P');
		}
		for(int i = 0; i < particlesRight.size(); i++)
		{
			drawObject(particlesRight[i].x, particlesRight[i].y, image2, 'P');
		}

		// CALCULATE HISTOGRAMS
		for(int i = 0; i < particlesLeft.size(); i++)
		{
			MatND histForParticle;
			Mat particleROI(image1, Rect(particlesLeft[i].x - 4, particlesLeft[i].y - 4, 8, 8));
			calcHist(&particleROI, 1, channels, Mat(), histForParticle, 2, histogramSize, ranges, true, false);
			particlesLeft[i].result = compareHist(histForParticle, patternHistogramLeft,CV_COMP_CORREL);
			//cout << particlesLeft[i].result << endl;
		}
		for(int i = 0; i < particlesRight.size(); i++)
		{
			MatND histForParticle;
			Mat particleROI(image2, Rect(particlesRight[i].x - 4, particlesRight[i].y - 4, 8, 8));
			calcHist(&particleROI, 1, channels, Mat(), histForParticle, 2, histogramSize, ranges, true, false);
			particlesRight[i].result = compareHist(histForParticle, patternHistogramRight,CV_COMP_CORREL);
		}

		// DELETE PARTICLES
		std::sort(particlesLeft.begin(), particlesLeft.end(), [](Particle a, Particle b){ return a.result < b.result; });
		for(int i = 0; i < particlesToDelete; i++)
			particlesLeft.erase(particlesLeft.begin());
		std::sort(particlesRight.begin(), particlesRight.end(), [](Particle a, Particle b){ return a.result < b.result; });
		for(int i = 0; i < particlesToDelete; i++)
			particlesRight.erase(particlesRight.begin());


		// ESTIMATE CENTER POINT AND SAVE FORMER
		formerEstimatedPositionLeft = estimatedPositionLeft; formerEstimatedPositionRight = estimatedPositionRight;
		estimatedPositionLeft.x = 0; estimatedPositionLeft.y = 0; estimatedPositionRight.x = 0; estimatedPositionRight.y = 0;
		for(int i = 0; i < particlesLeft.size(); i++)
		{
			estimatedPositionLeft.x += particlesLeft[i].x;
			estimatedPositionLeft.y += particlesLeft[i].y;
		}
		estimatedPositionLeft.x /= particlesLeft.size();
		estimatedPositionLeft.y /= particlesLeft.size();

		for(int i = 0; i < particlesRight.size(); i++)
		{
			estimatedPositionRight.x += particlesRight[i].x;
			estimatedPositionRight.y += particlesRight[i].y;
		}
		estimatedPositionRight.x /= particlesRight.size();
		estimatedPositionRight.y /= particlesRight.size();

		textStreamLeft << estimatedPositionLeft.x << " " << estimatedPositionLeft.y << "\n";
		textStreamRight << estimatedPositionRight.x << " " << estimatedPositionRight.y << "\n";

		// CALCULATE TRIANGULATED POINT AND SAVE
		Point3d result = IterativeLinearLSTriangulation(Point3d(estimatedPositionLeft.x, estimatedPositionLeft.y, 1.0), K1*RT1, Point3d(estimatedPositionRight.x, estimatedPositionRight.y, 1.0), K2*RT2);
		triangulatedTextStream << result.x << " " << result.y << " " << result.z << "\n";
		cout << result.x << " " << result.y << " " << result.z << "\n";

		// PERFORM KALMAN ESTIMATION AND SAVE
		Mat prediction = KF.predict();
		measurement(0) = result.x;
		measurement(1) = result.y;
		measurement(2) = result.z;
		Mat estimated = KF.correct(measurement);
		triangulatedKalmanTextStream << estimated.at<float>(0) << " " << estimated.at<float>(1) << " " << estimated.at<float>(2) << endl;

		// DRAW ESTIMATED POSITION
		drawObject(estimatedPositionLeft.x, estimatedPositionLeft.y, image1, 'C');
		drawObject(estimatedPositionRight.x, estimatedPositionRight.y, image2, 'C');

		// CALCULATE SPEED
		if(!firstSpeedEstimation)
		{
			leftSpeed.push_back(estimatedPositionLeft.x - formerEstimatedPositionLeft.x);
			leftSpeed.push_back(estimatedPositionLeft.y - formerEstimatedPositionLeft.y);
			rightSpeed.push_back(estimatedPositionRight.x - formerEstimatedPositionRight.x);
			rightSpeed.push_back(estimatedPositionRight.y - formerEstimatedPositionRight.y);
		}
		else
		{
			leftSpeed.push_back(0);
			leftSpeed.push_back(0);
			rightSpeed.push_back(0);
			rightSpeed.push_back(0);
		}

		// RANDOM NEW PARTICLES
		std::sort(particlesLeft.begin(), particlesLeft.end(), [](Particle a, Particle b){ return a.result > b.result; });
		vector<Particle> temp;
		
		for(int i = 0; i < 10; i++)
		{
			vector<Particle> _temp = randomNumberOfParticle(particlesNumber, rangeForRandom, image1.size().width, image1.size().height, 5, particlesLeft[i].x, particlesLeft[i].y, leftSpeed);
			temp.insert(temp.begin(), _temp.begin(), _temp.end());
		}
		for(int i = 10; i < 20; i++)
		{
			vector<Particle> _temp = randomNumberOfParticle(particlesNumber, rangeForRandom, image1.size().width, image1.size().height, 3, particlesLeft[i].x, particlesLeft[i].y, leftSpeed);
			temp.insert(temp.begin(), _temp.begin(), _temp.end());
		}
		for(int i = 20; i < 30; i++)
		{
			vector<Particle> _temp = randomNumberOfParticle(particlesNumber, rangeForRandom, image1.size().width, image1.size().height, 1, particlesLeft[i].x, particlesLeft[i].y, leftSpeed);
			temp.insert(temp.begin(), _temp.begin(), _temp.end());
		}
		particlesLeft.insert(particlesLeft.begin(), temp.begin(), temp.end());

		std::sort(particlesRight.begin(), particlesRight.end(), [](Particle a, Particle b){ return a.result > b.result; });
		temp.clear();
		
		for(int i = 0; i < 10; i++)
		{
			vector<Particle> _temp = randomNumberOfParticle(particlesNumber, rangeForRandom, image1.size().width, image1.size().height, 5, particlesRight[i].x, particlesRight[i].y, rightSpeed);
			temp.insert(temp.begin(), _temp.begin(), _temp.end());
		}
		for(int i = 10; i < 20; i++)
		{
			vector<Particle> _temp = randomNumberOfParticle(particlesNumber, rangeForRandom, image1.size().width, image1.size().height, 3, particlesRight[i].x, particlesRight[i].y, rightSpeed);
			temp.insert(temp.begin(), _temp.begin(), _temp.end());
		}
		for(int i = 20; i < 30; i++)
		{
			vector<Particle> _temp = randomNumberOfParticle(particlesNumber, rangeForRandom, image1.size().width, image1.size().height, 1, particlesRight[i].x, particlesRight[i].y, rightSpeed);
			temp.insert(temp.begin(), _temp.begin(), _temp.end());
		}
		particlesRight.insert(particlesRight.begin(), temp.begin(), temp.end());
		leftSpeed.clear();
		rightSpeed.clear();
		imshow("Particle filter - CAM1", image1);
		imshow("Particle filter - CAM2", image2);
		char key = waitKey(5);
		if (key == 'S')
			break;
	}
	cout << "End of video or user ended." << endl;
	system("pause");
	return 1;
}
