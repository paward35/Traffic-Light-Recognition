
#include "Utilities.h"
#include <iostream>
#include <fstream>
#include "opencv2/imgproc/imgproc.hpp"
using namespace std;

#define REGENTHOUSE_IMAGE_INDEX 0


Mat runkmeans(Mat testImage, int k, int iterations) {
	Mat samples(testImage.rows*testImage.cols, 3, CV_32F);
	float* sample = samples.ptr<float>(0);
	for (int row = 0; row < testImage.rows; row++)
		for (int col = 0; col < testImage.cols; col++)
			for (int channel = 0; channel < 3; channel++)
				samples.at<float>(row*testImage.cols + col, channel) =
				(uchar)testImage.at<Vec3b>(row, col)[channel];
	// Apply k-means clustering, determining the cluster
	// centres and a label for each pixel.
	Mat labels, centres;
	kmeans(samples, k, labels, TermCriteria(CV_TERMCRIT_ITER |
		CV_TERMCRIT_EPS, 0.0001, 10000), iterations,
		KMEANS_PP_CENTERS, centres);
	// Use centres and label to populate result image
	Mat& result_image = Mat(testImage.size(), testImage.type());
	for (int row = 0; row < testImage.rows; row++)
		for (int col = 0; col < testImage.cols; col++)
			for (int channel = 0; channel < 3; channel++)
				result_image.at<Vec3b>(row, col)[channel] =
				(uchar)centres.at<float>(*(labels.ptr<int>(
					row*testImage.cols + col)), channel);
	return result_image;
}

Mat opening(Mat testImage) {
	int erosion_type = MORPH_ELLIPSE;
	int erosion_size = 1;
	Mat element = getStructuringElement(erosion_type,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));
	Mat erodeResult;
	Mat  dilateResult;
	dilate(testImage, dilateResult, element);
	erode(dilateResult, erodeResult, element);
	return erodeResult;
}


Mat closing(Mat testImage) {
	int erosion_type = MORPH_ELLIPSE;
	int erosion_size = 1;
	Mat element = getStructuringElement(erosion_type,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));
	Mat erodeResult;
	Mat  dilateResult;
	erode(testImage, erodeResult, element);
	dilate(erodeResult, dilateResult, element);
	return erodeResult;
}

Mat runCanny(Mat imageToConvert) {
	Mat imageToConvert_gray;
	Mat dst, detected_edges;
	dst.create(imageToConvert.size(), imageToConvert.type());
	int lowThreshold = 0;
	const int max_lowThreshold = 150;
	const int ratio = 2;
	const int kernel_size = 3;

	cvtColor(imageToConvert, imageToConvert_gray, COLOR_BGR2GRAY);
	blur(imageToConvert_gray, detected_edges, Size(3, 3));
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
	dst = Scalar::all(0);
	imageToConvert_gray.copyTo(dst, detected_edges);
	return dst;
}

void ChamferMatchingz(Mat& chamfer_image, Mat& model, Mat& matching_image)
{
	// Extract the model points (as they are sparse).
	vector<Point> model_points;
	int image_channels = model.channels();
	for (int model_row = 0; (model_row < model.rows); model_row++)
	{
		uchar *curr_point = model.ptr<uchar>(model_row);
		for (int model_column = 0; (model_column < model.cols); model_column++)
		{
			if (*curr_point > 0)
			{
				Point& new_point = Point(model_column, model_row);
				model_points.push_back(new_point);
			}
			curr_point += image_channels;
		}
	}
	int num_model_points = model_points.size();
	image_channels = chamfer_image.channels();
	// Try the model in every possible position
	matching_image = Mat(chamfer_image.rows - model.rows + 1,
		chamfer_image.cols - model.cols + 1, CV_32FC1);
	for (int search_row = 0; (search_row <=
		chamfer_image.rows - model.rows); search_row++)
	{
		float *output_point = matching_image.ptr<float>(search_row);
		for (int search_column = 0; (search_column <=
			chamfer_image.cols - model.cols); search_column++)
		{
			float matching_score = 0.0;
			for (int point_count = 0; (point_count < num_model_points);
				point_count++)
				matching_score += (float) *(chamfer_image.ptr<float>(
					model_points[point_count].y + search_row) + search_column +
					model_points[point_count].x*image_channels);
			*output_point = matching_score;
			output_point++;
		}
	}
}


void CompareRecognitionResult(vector<vector<Point>>  locations_found, int** GroundTruth, double& precision, double& recall, double& accuracy, double& specificity, double& f1)
{
	int false_positives = 0;
	int false_negatives = 0;
	int true_positives = 0;
	int true_negatives = 0;
	Point pt1;
	Point pt2;

	for (int pictureIndex = 0; pictureIndex < locations_found.size() ; pictureIndex++) {
		for (int col = 0; col < locations_found[pictureIndex].size(); col = col + 2)
		{
			bool foundTruePos = false;
			
			for (int gtLightIndex = 4; gtLightIndex < *GroundTruth[pictureIndex]; gtLightIndex = gtLightIndex + 7) {
				pt1.x = GroundTruth[pictureIndex][gtLightIndex];
				pt1.y = GroundTruth[pictureIndex][gtLightIndex + 1];
				pt2.x = GroundTruth[pictureIndex][gtLightIndex + 2];
				pt2.y = GroundTruth[pictureIndex][gtLightIndex + 3];

				Point foundPt1 = locations_found[pictureIndex][col];
				Point foundPt2 = locations_found[pictureIndex][col+1];
				Rect found = Rect(foundPt1, foundPt2);
				Rect groundTruth = Rect(pt1, pt2);
				bool intersection = ((found & groundTruth).area() > (found.area() * .8) );
				if (intersection) {
					foundTruePos = true;
				}


			}

			if (foundTruePos) {
				true_positives++;
				foundTruePos = false;
			}
			else {
				false_positives++;
			}
		}
		

	}
	false_negatives = 30 - true_positives;
	cout << "\n true_positives: " << true_positives;
	cout << "\n false_positives: " << false_positives;
	cout << "\n false_negatives: " << false_negatives;
	cout << "\n";
	precision = ((double)true_positives) / ((double)(true_positives + false_positives));
	recall = ((double)true_positives) / ((double)(true_positives + false_negatives));
	accuracy = ((double)(true_positives + true_negatives)) / ((double)(true_positives + false_positives + true_negatives + false_negatives));
	specificity = ((double)true_negatives) / ((double)(false_positives + true_negatives));
	f1 = 2.0*precision*recall / (precision + recall);
}



int main(int argc, const char** argv)
{
	int** GroundTruth;
	GroundTruth = new int*[14];
	//Load images and ground truth
	Mat testImages[14];
	testImages[0] = imread("Media/CamVidLights01.png", CV_LOAD_IMAGE_COLOR);
	int row0[15] = {15, 0,0,1, 319,202,346,279,0,0,1, 692,264,711,322 };
	GroundTruth[0] = row0;

	testImages[1] = imread("Media/CamVidLights02.png", CV_LOAD_IMAGE_COLOR);
	int row1[15] = {15, 0,0,1, 217,103,261,230, 0,0,1, 794,212,820,294};
	GroundTruth[1] = row1;

	testImages[2] = imread("Media/CamVidLights03.png", CV_LOAD_IMAGE_COLOR);
	int row2[15] = { 15, 0,0,1, 347,210,373,287, 0,0,1,701,259,720,318 };
	GroundTruth[2] = row2;

	testImages[3] = imread("Media/CamVidLights04.png", CV_LOAD_IMAGE_COLOR);
	int row3[15] = { 15, 0,0,1, 271,65,339,191,1,0,0, 640,260,662,301 };
	GroundTruth[3] = row3;

	testImages[4] = imread("Media/CamVidLights05.png", CV_LOAD_IMAGE_COLOR);
	int row4[15] = { 15, 1,1,0, 261,61,333,195,1,1,0, 644,269,666,313 };
	GroundTruth[4] = row4;

	testImages[5] = imread("Media/CamVidLights06.png", CV_LOAD_IMAGE_COLOR);
	int row5[15] = { 15, 0,0,1, 238,42,319,190,0,0,1, 650,279,672,323 };
	GroundTruth[5] = row5;

	testImages[6] = imread("Media/CamVidLights07.png", CV_LOAD_IMAGE_COLOR);
	int row6[15] = { 15, 0,1,0, 307,231,328,297,0,1,0, 747,266,764,321 };
	GroundTruth[6] = row6;

	testImages[7] = imread("Media/CamVidLights08.png", CV_LOAD_IMAGE_COLOR);
	int row7[15] = { 15, 0,1,0, 280,216,305,296,0,1,0, 795,253,816,316 };
	GroundTruth[7] = row7;

	testImages[8] = imread("Media/CamVidLights09.png", CV_LOAD_IMAGE_COLOR);
	int row8[15] = { 15, 0,0,1, 359,246,380,305,0,0,1, 630,279,646,327 };
	GroundTruth[8] = row8;

	testImages[9] = imread("Media/CamVidLights10.png", CV_LOAD_IMAGE_COLOR);
	int row9[15] = { 15, 0,0,1, 260,122,299,239,0,0,1, 691, 271,714,315 };
	GroundTruth[9] = row9;

	testImages[10] = imread("Media/CamVidLights11.png", CV_LOAD_IMAGE_COLOR);
	int row10[15] = { 15, 0,0,1, 331,260,349,312,0,0,1, 663,280,676,322 };
	GroundTruth[10] = row10;

	testImages[11] = imread("Media/CamVidLights12.png", CV_LOAD_IMAGE_COLOR);
	int row11[29] = { 29, 0,0,1, 373,219,394,279,0,0,1, 715,242,732,299,
		1,0,0, 423,316,429,329,1,0,0, 516,312,521,328 };
	GroundTruth[11] = row11;

	testImages[12] = imread("Media/CamVidLights13.png", CV_LOAD_IMAGE_COLOR);
	int row12[15] = { 15, 1,0,0, 283,211,299,261,1,0,0, 604,233,620,279 };
	GroundTruth[12] = row12;

	testImages[13] = imread("Media/CamVidLights14.png", CV_LOAD_IMAGE_COLOR);
	int row13[15] = { 15, 0,0,1, 294,188,315,253,0,0,1, 719,225,740,286 };
	GroundTruth[13] = row13;
	cv::waitKey(0);

	//Declare Varibles to use in loop
	RNG rng(12345);
	Scalar colorForContours = Scalar(200, 50, 50);
	Scalar colorForBoundingBoxes = Scalar(50, 200, 50);
	Scalar ColorForLights = Scalar(50, 50, 200);
	//Points to draw ground truth
	Point pt1;
	Point pt2;
	//Colour to draw ground truth 
	const Scalar& color = Scalar(0, 0, 255);
	//Other variables to draw ground truth 
	int thickness = 1;
	int lineType = 8;
	int shift = 0;

	//Vectors for results
	vector<vector<Point>> lightBoxsFound (14);
	vector<vector<int>> lightState(14);
	
	


	//Loop through all of the images
	for (int j = 5; j < sizeof(testImages) / sizeof(testImages[0]); j++) {
	//for (int j = 10; j < 12; j++) {
		//Declare images for conversion
		Mat imageToConvert_gray, converted_image, contour_image;
		//Convert images to grayscale 
		cvtColor(testImages[j], imageToConvert_gray, COLOR_BGR2GRAY);
		//Set Variables for thresholding 
		int threshold_value = 58; //50 worked well
		int max_BINARY_value = 255;
		//Threshold Image
		threshold(imageToConvert_gray, converted_image, threshold_value, max_BINARY_value, 1);
		//Create varibles for coutours 
		vector<vector<Point>> contours;
		//Create vector to hold contour hierarchy 
		vector<Vec4i> hierarchy;

		findContours(converted_image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));

		//Create variables to hold countour shape information 
		vector<vector<Point> > contours_poly(contours.size());
		vector<vector<Point> > hulls(contours.size());
		vector<vector<int> > lightsFromContours(contours.size());
		vector<Rect> boundRect(contours.size());
		vector<Point2f>center(contours.size());
		vector<float>radius(contours.size());
		vector<RotatedRect> minEllipse(contours.size());


		// Approximate contours to polygons + get bounding rects and circles
		for (int i = 0; i < contours.size(); i++)
		{
			
			//Tempary rectangle for bounding box 
			Rect tempTrafficBox;
			//Create shape from countours for drawing
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			//Create bounding rectangle for coutour 
			tempTrafficBox = boundingRect(Mat(contours[i]));
			//Get height an width of contour 
			double recWidth = (double)tempTrafficBox.width;
			double recHeight = (double)tempTrafficBox.height;
			//Caculate aspect ratio and area for countors 
			double ratio = recWidth / recHeight;
			int area = tempTrafficBox.width * tempTrafficBox.height;
			double actualArea = contourArea(contours_poly[i]);
			//Only consider countours that meet these critera 
			
			convexHull(contours_poly[i], hulls[i],false);
			double convexArea = contourArea(hulls[i]);
			double solidarity = actualArea / convexArea;
			//
		

		
			//if ((ratio > .25) && (ratio < .8) && (area > 350) && (area < 13000) && ((actualArea / area) > .65)  ) {
			if ((area > 520) && (area < 12000) && (ratio > .25) && (ratio < .8) && ((actualArea / area) > .52) && (solidarity > .7) ) {
				vector<int> lights = { 0,0,0 };
				bool foundLight = false;
				//Get child countour of bounding box
				int childContour = hierarchy[i][2];
				//For all contours on the same level draw a circle around the coutour
				while (childContour != -1) {
					
					//Create shape from countours for drawing
					if (contours[childContour].size() > 5)
					{
						RotatedRect tempEllipse = fitEllipse(Mat(contours[childContour]));
						
						double minorAxis = tempEllipse.size.height /2;
						double majorAxis = tempEllipse.size.width /2;
						double ellipse_area = minorAxis * majorAxis;
						double c = sqrt(minorAxis*minorAxis + majorAxis*majorAxis);
						double e = minorAxis / c;
						//Check how circularity and size of ellipse 
						if (e < .90 &&  ellipse_area > 10) {
							minEllipse[childContour] = tempEllipse;
							//If ellipse is circular then found light is true
							foundLight = true;
							//Record ellipse to draw later
							minEllipse[childContour] = tempEllipse;
							//Get center of ellipse to check position of ellipse
							float centerx = tempEllipse.center.x;
							float centery = tempEllipse.center.y;
							//Get center of light to check position of rectangle 
							Point center_of_rect = (tempTrafficBox.br() + tempTrafficBox.tl())*0.5;
				
							if (center_of_rect.y < centery - 10) {
								lights[2] = 1;
							}else if ((center_of_rect.y > centery + 10)) {
								lights[0] = 1;
							} else{
								lights[1] = 1;
							}
						}
						
						
					}
				
					childContour = hierarchy[childContour][0];
				}
				//If box has child countour add the rectangle to the images to be drawn 
				if (hierarchy[i][2] != -1 && foundLight) {
					foundLight = false;
					lightsFromContours[i] = lights;
					boundRect[i] = boundingRect(Mat(contours_poly[i]));
				}
			}
		}
		int count = 0;
		

		//Draw contour image
		Mat drawing = Mat::zeros(converted_image.size(), CV_8UC3);
		for (int i = 0; i < contours.size(); i++)
		{
			
			if (! lightsFromContours[i].empty()) {
				//Capture results for analysis
				lightBoxsFound[j].push_back(boundRect[i].tl());
				lightBoxsFound[j].push_back(boundRect[i].br());
				String textToPrint = "";
				for (int p = 0; p < 3; p++) {
					lightState[j].push_back(lightsFromContours[i][p]);
					if (lightsFromContours[i][p]) {
						if (p == 2) {
							textToPrint += " Green ";
						}
						else if (p == 1) {
							textToPrint += " Amber ";
						}
						else if (p == 0) {
							textToPrint += " Red ";
						}
					}
					//put text on image
					putText(drawing, textToPrint, boundRect[i].br(),
						FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 250, 50), 1, CV_AA);
				}
			}
			//draw contours, rectangle and ellipses
			drawContours(drawing, contours_poly, i, colorForContours, 1, 8, vector<Vec4i>(), 0, Point());
			cv::rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), colorForBoundingBoxes, thickness, 8, 0);
			cv::ellipse(drawing, minEllipse[i], color, 2, 8);
		}

		//Draw ground truth
		for (int k = 4; k < *GroundTruth[j]; k = k + 7) {
			pt1.x = GroundTruth[j][k];
			pt1.y = GroundTruth[j][k + 1];
			pt2.x = GroundTruth[j][k + 2];
			pt2.y = GroundTruth[j][k + 3];
			cv::rectangle(testImages[j], pt1, pt2, color, thickness, lineType, shift);
		}

		//Show gray image
		//cv::imshow("GrayScale Image : " + std::to_string(j), imageToConvert_gray);
		//Show thresholded image 
		//cv::imshow("Thesholded Image: " + std::to_string(j), converted_image);
		//Show Contour image
		cv::imshow("Contour image: " + std::to_string(j + 1), drawing);
	

		
		std::cout << "Done!";
		
		

		//cv::destroyAllWindows();
		
	}
	std::cout << "Done!";
	double precision;
	double  recall;
	double  accuracy;
	double  specificity;
	double  f1;
	//Calculate performance metrics
	CompareRecognitionResult(lightBoxsFound, GroundTruth, precision , recall, accuracy, specificity, f1);
	cv::imshow("Contour image: " , testImages[12]);

	std::cout << "\precision:" << precision;
	std::cout << "\nrecall: " <<   recall;
	std::cout << "\naccuracy: " <<   accuracy;
	std::cout << "\nspecificity: " <<   specificity;
	std::cout << "\nf1" <<   f1;
	cvWaitKey(0);


	

	cv::destroyAllWindows();
	cvWaitKey(0);


}



