#pragma once
//TODO everything more or less
namespace ScanMoCap {
	const unsigned int width = 640;
	const unsigned int widthL2 = unsigned int(0.5f * float(width));
	const unsigned int widthL3 = unsigned int(0.5f * float(widthL2));

	const unsigned int widthXYZ = 3 * width;
	const unsigned int widthL2XYZ = 3 * widthL2;
	const unsigned int widthL3XYZ = 3 * widthL3;

	const unsigned int height = 480;
	const unsigned int heightL2 = unsigned int(0.5f * float(height));
	const unsigned int heightL3 = unsigned int(0.5f * float(heightL2));

	const unsigned int pixelCount = width * height;
	const unsigned int pixelCountL2 = unsigned int(0.25f * float(pixelCount));
	const unsigned int pixelCountL3 = unsigned int(0.25f * float(pixelCountL2));

	const unsigned int pixelCountXYZ = widthXYZ * height;
	const unsigned int pixelCountXYZL2 = unsigned int(0.25f * float(pixelCountXYZ));
	const unsigned int pixelCountXYZL3 = unsigned int(0.25f * float(pixelCountXYZL2));

	const unsigned int maxDepth = 10000; 
	const unsigned int minDepth = 0; 
}
