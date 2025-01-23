#pragma once
namespace ScanMoCap {
	const unsigned int pixelCount = 76800; //TODO Also defined in "this"...
	const unsigned int pixelCountL2 = unsigned int(0.25f * float(pixelCount));
	const unsigned int pixelCountL3 = unsigned int(0.25f * float(pixelCountL2));

	const unsigned int pixelCountXYZ = 3 * 76800;
	const unsigned int pixelCountXYZL2 = unsigned int(0.25f * float(pixelCountXYZ));
	const unsigned int pixelCountXYZL3 = unsigned int(0.25f * float(pixelCountXYZL2));

	const unsigned int stride = 3;

	const unsigned int width = 320;
	const unsigned int widthL2 = unsigned int(0.5f * float(width));
	const unsigned int widthL3 = unsigned int(0.5f * float(widthL2));

	const unsigned int widthXYZ = 3 * width;
	const unsigned int widthL2XYZ = 3 * widthL2;
	const unsigned int widthL3XYZ = 3 * widthL3;

	const unsigned int height = 240;
	const unsigned int heightL2 = unsigned int(0.5f * float(height));
	const unsigned int heightL3 = unsigned int(0.5f * float(heightL2));
}
