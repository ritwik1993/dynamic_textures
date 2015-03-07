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
	int width = frames[0].size().width;
	int height = frames[0].size().height;
	int m = width * height;
	int tau = frames.size();

	
	// --- Average the pixels of all frames --- //
	
	// Ymean = mean(Y,2);
	cv::Mat Y;
	convertFramesToVectors( frames, Y );
	
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
	
	cv::Mat temp = Y - (Ymean*cv::Mat::ones(1,tau, CV_32FC3));
	// SVD HERE:  [U,S,V] = svd(temp, 0);
	// U should be m x n
	// S should be n x n diagonal
	// V should be tau x n

	// DELETE THIS WHEN SVD WORKS, THIS IS JUST TO GET CODE TO COMPILE
	cv::Mat U( m, n, CV_32FC3 );
	cv::Mat S( n, n, CV_32FC3 );
	cv::Mat V( tau, n, CV_32FC3 );
	// END DELETE

	Chat = cv::Mat( U );

	cv::Mat Xhat( S*(V.t()) );

	//cv::Mat x0 = Xhat(:,1);
	x0 = cv::Mat( n, 1, CV_32FC3 );
	for( int i = 0; i < n; ++i )
	{
		cv::Vec3f& v = x0.at<cv::Vec3f>(i, 0);
		const cv::Vec3f& xhatPixel = Xhat.at<cv::Vec3f>(i, 0);
		v[0] = xhatPixel[0];
		v[1] = xhatPixel[1];
		v[2] = xhatPixel[2];
	}
	
	// Ahat = Xhat(:,2:tau)*pinv(Xhat(:,1:(tau-1)));

	// Vhat = Xhat(:,2:tau)-Ahat*Xhat(:,1:(tau-1));

	// [Uv,Sv,Vv] = svd(Vhat,0);

	// Bhat = Uv(:,1:nv)*Sv(1:nv,1:nv)/sqrt(tau-1);
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
cv::Mat dummy;
learnDynamicTexture( frames, 0, 0,
					 dummy, dummy,
					 dummy, dummy, dummy );	


	playVideoSequence( frames );
	
	return 0;
}
