#pragma once
#include <OpenNI.h>
#include <iostream>
#include "PrimeSenseVideoStreamListener.h"
#include <queue>
#include "Structs.h"
#include <memory>

using namespace openni;
using namespace std;

class PrimeSenseConnectedListener : public OpenNI::DeviceConnectedListener
{
public:
	function<void(unique_ptr<Device>)> callbackFunc;

	PrimeSenseConnectedListener(function<void(unique_ptr<Device>)> callback) : callbackFunc(move(callback)) {};

	void onDeviceConnected(const DeviceInfo* deviceInfo) override
	{
		unique_ptr<Device> device = make_unique<Device>();

		cout << deviceInfo->getVendor() << std::endl;
		cout << deviceInfo->getName() << std::endl;
		cout << deviceInfo->getUsbProductId() << std::endl;
		cout << deviceInfo->getUsbVendorId() << std::endl;

		//TODO enable
		/*
		if (deviceInfo->getVendor() != "PrimeSense")
		{
			cout << "Found unrelated device." << endl;
			return;
		}
		*/

		auto status = device->open(deviceInfo->getUri());

		if (status != Status::STATUS_OK)
		{
			cerr << "Device could not be opened! " << status << endl;
			return;
		}

		status = device->setDepthColorSyncEnabled(true);

		if (status != Status::STATUS_OK)
		{
			cout << "No Depth Color Sync Possible! " << status << endl;
		}

		if (device->isImageRegistrationModeSupported(ImageRegistrationMode::IMAGE_REGISTRATION_DEPTH_TO_COLOR))
		{
			status = device->setImageRegistrationMode(ImageRegistrationMode::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
			cout << "Registration Mode Depth2Color: " << status << endl;
		}
		else
		{
			cout << "No Registration Mode supported.";
		}

		callbackFunc(move(device));
	}
};
