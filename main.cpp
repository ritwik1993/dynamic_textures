#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>


void printMatrixSize( const cv::Mat& matrix, const std::string& matName )
{
	std::cout << matName << " size:  " << matrix.size().height << " x " << matrix.size().width << std::endl;
}


// Takes in the location of a video and a container for video frames and fills
// the container with the frames from the video.  Returns true if successulf,
// false otherwise.

bool readVideoFrames( const std::string& videoLoc, std::vector<cv::Mat>& frames )
{
	// http://projects.cwi.nl/dyntex/database.html
	cv::VideoCapture cap(videoLoc);

	if( !cap.isOpened() )
	{
		std::cout << "Cannot open the video file!" << std::endl;
		return false;
	}

	while(frames.size() < 100)
	{
		cv::Mat frame;
		
		// Read a new frame from the video.
		if( !cap.read(frame) )
			break;

		// TODO:  CONVERT FRAME TO CV_32FC3 FOR PROCESSING BELOW
		
		// For now read in only a portion of the data so that we can test faster.
		frames.push_back(frame.rowRange(0, 50).colRange(0, 50));
	}

	std::cout << "Num frames:  " << frames.size() << std::endl;
	printMatrixSize( frames[0], "Frame" );
	std::cout << "Type:  " << frames[0].type() << std::endl;
	if( frames[0].type() == CV_8UC3 )
		std::cout << "Type is CV_8UC3" << std::endl;
	
	return true;
}


// Takes in a container of video frames and plays them back assuming 30 Hz playback.

void playVideoSequence( const std::vector<cv::Mat>& frames )
{
	cv::namedWindow("Video", CV_WINDOW_AUTOSIZE);

	for( std::size_t i = 0; i < frames.size(); ++i )
	{
		cv::imshow("Video", frames[i]);

		if( cv::waitKey(33) == 27 )
		{
			std::cout << "ESC key is pressed by user" << std::endl;
			break;
		}
	}
}


void convertFramesToVectors( const std::vector<cv::Mat>& frames, cv::Mat& Y1, cv::Mat& Y2, cv::Mat& Y3 )
{
	int width = frames[0].size().width;
	int height = frames[0].size().height;
	int m = width*height;
	int tau = frames.size();
	Y1 = cv::Mat(m, tau, CV_32F);
	Y2 = cv::Mat(m, tau, CV_32F);
	Y3 = cv::Mat(m, tau, CV_32F);
	
	for( std::size_t i = 0; i < frames.size(); ++i )
	{
		const cv::Mat& frame = frames[i];

		for( int row = 0; row < height; ++row )
		{
			for( int col = 0; col < width; ++col )
			{
				const cv::Vec3b& oldPixel = frame.at<cv::Vec3b>(row, col);

				Y1.at<cv::Vec3f>( row*width+col , i ) = static_cast<float>(oldPixel[0]);
				Y2.at<cv::Vec3f>( row*width+col , i ) = static_cast<float>(oldPixel[1]);
				Y3.at<cv::Vec3f>( row*width+col , i ) = static_cast<float>(oldPixel[2]);
			}
		}
	}
}


void convertVectorsToFrames( const cv::Mat& vectors, std::vector<cv::Mat>& frames,
							 int width, int height )
{
	int m = vectors.size().height;
	int tau = vectors.size().width;

	frames.clear();

	for( int v = 0; v < tau; ++v )
	{
		frames.push_back( cv::Mat( height, width, CV_32FC3 ) );
		cv::Mat& frame = frames[v];
		
		for( int p = 0; p < m; ++p )
		{
			int row = p / width;
			int col = p % width;

			const cv::Vec3f& vecPixel = vectors.at<cv::Vec3f>( p, v );
			cv::Vec3f& framePixel = frame.at<cv::Vec3f>( row, col );
			framePixel[0] = vecPixel[0];
			framePixel[1] = vecPixel[1];
			framePixel[2] = vecPixel[2];
		}
	}
}


void learnDynamicTexture( const std::vector<cv::Mat>& frames, int n, int nv,
						  cv::Mat& x0, cv::Mat& Y1mean, cv::Mat& Y2mean, cv::Mat& Y3mean,
						  cv::Mat& Ahat, cv::Mat& Bhat, cv::Mat& Chat )
{
	std::cout << "In learnDynamicTexture" << std::endl;
	int width = frames[0].size().width;
	int height = frames[0].size().height;
	int m = width * height;
	int tau = frames.size();

	std::cout << std::endl;
	std::cout << "tau:  " << tau << std::endl;
	std::cout << "m:  " << m << std::endl;
	std::cout << "n:  " << n << std::endl;
	std::cout << "nv:  " << nv << std::endl;
	std::cout << std::endl;
	
	// --- Average the pixels of all frames --- //

	std::cout << "Calling convertFramesToVectors" << std::endl;
	// Ymean = mean(Y,2);
	cv::Mat Y1;
	cv::Mat Y2;
	cv::Mat Y3;
	convertFramesToVectors( frames, Y1, Y2, Y3 );

	printMatrixSize( Y1, "Y1" );
	printMatrixSize( Y2, "Y2" );
	printMatrixSize( Y3, "Y3" );

	std::cout << "Calculating the averages" << std::endl;
	
	Y1mean = cv::Mat(m, 1, CV_32F);
	Y2mean = cv::Mat(m, 1, CV_32F);
	Y3mean = cv::Mat(m, 1, CV_32F);

	for( int row = 0; row < height; ++row )
	{
		float& f1 = Y1mean.at<float>(row, 0);
		float& f2 = Y2mean.at<float>(row, 0);
		float& f3 = Y3mean.at<float>(row, 0);
		f1 = 0;
		f2 = 0;
		f3 = 0;

		for( int col = 0; col < width; ++col )
		{
			f1 += Y1.at<float>(row,col);
			f2 += Y2.at<float>(row,col);
			f3 += Y3.at<float>(row,col);
		}

		f1 /= static_cast<float>(width);
		f2 /= static_cast<float>(width);
		f3 /= static_cast<float>(width);
	}


	// [U,S,V] = svd(Y-Ymean*ones(1,tau),0);
	std::cout << "Creating ones" << std::endl;
	cv::Mat ones = cv::Mat::ones(1, tau, CV_32F);
	std::cout << "Y1mean * ones" << std::endl;
	cv::Mat temp = Y1mean * ones;
	std::cout << "Y1 - Y1mean" << std::endl;
	temp = Y1 - temp;
	
	std::cout << "Calculating first SVD" << std::endl;
	printMatrixSize( temp, "temp" );
	
	cv::SVD svd_temp( temp );

	printMatrixSize( svd_temp.u, "svd_temp.u" );
	printMatrixSize( svd_temp.w, "svd_temp.w" );
	printMatrixSize( svd_temp.vt, "svd_temp.vt" );


	cv::Mat sigma = cv::Mat::zeros( n, n, CV_32F );
	for( int i = 0; i < n; ++i )
	{
		sigma.at<float>(i, i) = svd_temp.w.at<float>(i, 0);
	}

	printMatrixSize( sigma, "Sigma" );

	cv::Mat VT = svd_temp.vt.rowRange(0, n);

	printMatrixSize( VT, "VT" );
	
	cv::Mat U = svd_temp.u.colRange(0, n);

	printMatrixSize( U, "U" );

	std::cout << "Second" << std::endl;
	Chat = U;
	printMatrixSize( Chat, "Chat" );

	// V` <- does this equal V.t()?  Matlab code mentions Hermition transpose...
	//cv::Mat Xhat( S.colRange(0, n-1).rowRange(0, n-1) * (V.colRange(0,n-1)) );
	std::cout << "Third" << std::endl;
	
	cv::Mat Xhat( sigma * VT );
	printMatrixSize( Xhat, "Xhat" );

	std::cout << "Fourth" << std::endl;
	x0 = Xhat.col(0);
	printMatrixSize( x0, "x0" );

	std::cout << "Fifth" << std::endl;
	Ahat = Xhat.colRange(1, tau-1) * (Xhat.colRange(0, tau-2).inv(cv::DECOMP_SVD));
	printMatrixSize( Ahat, "Ahat" );

	std::cout << "Sixth" << std::endl;
	cv::Mat Vhat = Xhat.colRange(1, tau-1) - (Ahat * (Xhat.colRange(0, tau-2)));
	printMatrixSize( Vhat, "Vhat" );

	// [Uv,Sv,Vv] = svd(Vhat,0);
	std::cout << "Seventh" << std::endl;
	cv::SVD svd_Vhat( Vhat );

	//Bhat = U.colRange(0, nv-1) * Sv.rowRange(0, nv-1).colRange(0, nv-1);
	std::cout << "Eighth" << std::endl;
	Bhat = U * sigma;
	std::cout << "Ninth" << std::endl;
	Bhat /= sqrt(tau-1);
	printMatrixSize( Bhat, "Bhat" );
	std::cout << "Exit" << std::endl;
}


cv::Mat synthesizeFrame( const cv::Mat& x0, const cv::Mat& Ymean, const cv::Mat& Ahat, const cv::Mat& Bhat, const cv::Mat& Chat, double tau )
{
	int n = Bhat.size().height;
	int nv = Bhat.size().width;

	
	
	return cv::Mat();
}


int main( int argc, char** argv )
{
	std::cout << "Reading video file." << std::endl;
	
	std::vector<cv::Mat> frames;
	if( !readVideoFrames("./645c620.avi", frames) )
	{
		std::cout << "Could not read video file!" << std::endl;
		return -1;
	}

	std::cout << "Calling learnDynamicTexture" << std::endl;

	cv::Mat x0, Y1mean, Y2mean, Y3mean, Ahat, Bhat, Chat;
	learnDynamicTexture( frames, 25, 20,
						 x0, Y1mean, Y2mean, Y3mean,
						 Ahat, Bhat, Chat );

	std::cout << "Exited learnDynamicTexture" << std::endl;

	playVideoSequence( frames );
	
	return 0;
}
