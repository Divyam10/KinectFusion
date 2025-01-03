#pragma once
struct ImageData
{
	const uint8_t* colorData;
	const uint16_t* depthData;

	ImageData(const uint8_t* color, const uint16_t* depth)
		: colorData(color), depthData(depth) {
	}
};
