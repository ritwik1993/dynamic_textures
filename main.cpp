#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>


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

	while(1)
	{
		cv::Mat frame;
		
		// Read a new frame from the video.
		if( !cap.read(frame) )
			break;

		// TODO:  CONVERT FRAME TO CV_32FC3 FOR PROCESSING BELOW

		frames.push_back(frame);
	}

	std::cout << "Num frames:  " << frames.size() << std::endl;
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


void convertFramesToVectors( const std::vector<cv::Mat>& frames, cv::Mat& Y )
{
	int width = frames[0].size().width;
	int height = frames[0].size().height;
	int m = width*height;
	int tau = frames.size();
	Y = cv::Mat(m, tau, CV_32FC3);

	for( std::size_t i = 0; i < frames.size(); ++i )
	{
		const cv::Mat& frame = frames[i];

		for( int row = 0; row < height; ++row )
		{
			for( int col = 0; col < width; ++col )
			{
				const cv::Vec3b& oldPixel = frame.at<cv::Vec3b>(row, col);

				cv::Vec3f& newPixel = Y.at<cv::Vec3f>( row*width+col , i );

				newPixel[0] = static_cast<float>(oldPixel[0]);
				newPixel[1] = static_cast<float>(oldPixel[1]);
				newPixel[2] = static_cast<float>(oldPixel[2]);
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
						  cv::Mat& x0, cv::Mat& Ymean,
						  cv::Mat& Ahat, cv::Mat& Bhat, cv::Mat& Chat )
{
	std::cout << "In learnDynamicTexture" << std::endl;
	int width = frames[0].size().width;
	int height = frames[0].size().height;
	int m = width * height;
	int tau = frames.size();

	
	// --- Average the pixels of all frames --- //

	std::cout << "Calling convertFramesToVectors" << std::endl;
	// Ymean = mean(Y,2);
	cv::Mat Y;
	convertFramesToVectors( frames, Y );

	std::cout << "Calculating the average" << std::endl;
	
	Ymean = cv::Mat(m, 1, CV_32FC3);

	for( int row = 0; row < height; ++row )
	{
		cv::Vec3f& meanPixel = Ymean.at<cv::Vec3f>(row, 0);
		meanPixel[0] = 0;
		meanPixel[1] = 0;
		meanPixel[2] = 0;

		for( int col = 0; col < width; ++col )
		{
			const cv::Vec3f& p = Y.at<cv::Vec3f>(row, col);

			meanPixel[0] += p[0];
			meanPixel[1] += p[1];
			meanPixel[2] += p[2];
		}

		meanPixel /= static_cast<float>(width);
	}


	// --- Singular Value Decomposition --- //

	std::cout << "Singular value decomposition" << std::endl;

	if( Y.type() == CV_32FC3 )
		std::cout << "\tY type is CV_32FC3" << std::endl;

	if( Ymean.type() == CV_32FC3 )
		std::cout << "\tYmean type is CV_32FC3" << std::endl;

	std::cout << "Creating ones" << std::endl;
	cv::Mat ones = cv::Mat::ones(1, tau, CV_32FC3);
	std::cout << "Ymean * ones" << std::endl;
	cv::Mat temp = Ymean * ones;
	std::cout << "Y - Ymean" << std::endl;
	temp = Y - temp;
	
	
		//cv::Mat temp = Y - (Ymean * cv::Mat::ones(1,tau, CV_32FC3));
	std::cout << "First" << std::endl;
	cv::SVD svd_temp( temp );
	// SVD HERE:  [U,S,V] = svd(temp, 0);
	// U should be m x n
	// S should be n x n diagonal
	// V should be tau x n

	// DELETE THIS WHEN SVD WORKS, THIS IS JUST TO GET CODE TO COMPILE
	// cv::Mat U( m, n, CV_32FC3 );
	// cv::Mat S( n, n, CV_32FC3 );
	// cv::Mat V( tau, n, CV_32FC3 );
	// END DELETE


	// TEST THE SIZES OF THE MATRICES.
	
	
	//Chat = U.colRange(0, n-1);
	std::cout << "Second" << std::endl;
	Chat = svd_temp.u;

	// V` <- does this equal V.t()?  Matlab code mentions Hermition transpose...
	//cv::Mat Xhat( S.colRange(0, n-1).rowRange(0, n-1) * (V.colRange(0,n-1)) );
	std::cout << "Third" << std::endl;
	cv::Mat Xhat( svd_temp.w * svd_temp.vt );

	std::cout << "Fourth" << std::endl;
	x0 = Xhat.col(0);

	std::cout << "Fifth" << std::endl;
	Ahat = Xhat.colRange(1, tau-1) * (Xhat.colRange(0, tau-2).inv(cv::DECOMP_SVD));

	std::cout << "Sixth" << std::endl;
	cv::Mat Vhat = Xhat.colRange(1, tau-1) - (Ahat * (Xhat.colRange(0, tau-2)));

	// [Uv,Sv,Vv] = svd(Vhat,0);
	std::cout << "Seventh" << std::endl;
	cv::SVD svd_Vhat( Vhat );

	//Bhat = U.colRange(0, nv-1) * Sv.rowRange(0, nv-1).colRange(0, nv-1);
	std::cout << "Eighth" << std::endl;
	Bhat = svd_Vhat.u * svd_Vhat.w;
	std::cout << "Ninth" << std::endl;
	Bhat /= sqrt(tau-1);
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

	
	// DUMMY TEST
	std::cout << "Calling learnDynamicTexture" << std::endl;
cv::Mat dummy1;
cv::Mat dummy2;
cv::Mat dummy3;
cv::Mat dummy4;
cv::Mat dummy5;
learnDynamicTexture( frames, 0, 0,
					 dummy1, dummy2,
					 dummy3, dummy4, dummy5 );	
std::cout << "Exited learnDynamicTexture" << std::endl;

	playVideoSequence( frames );
	
	return 0;
}
