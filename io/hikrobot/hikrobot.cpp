#include "hikrobot.hpp"

#include <libusb-1.0/libusb.h>

#include <cstring>
#include <mutex>

#include "tools/logger.hpp"

using namespace std::chrono_literals;

namespace
{
std::mutex & hikrobot_sdk_mutex()
{
  static std::mutex mutex;
  return mutex;
}

std::string usb_serial(const MV_CC_DEVICE_INFO * info)
{
  if (info == nullptr || info->nTLayerType != MV_USB_DEVICE) return {};
  const char * text = reinterpret_cast<const char *>(info->SpecialInfo.stUsb3VInfo.chSerialNumber);
  return std::string(text, strnlen(text, INFO_MAX_BUFFER_SIZE));
}

}  // namespace

namespace io
{
HikRobot::HikRobot(
  double exposure_ms, double gain, const std::string & vid_pid, const std::string & serial)
: exposure_us_(exposure_ms * 1e3),
  gain_(gain),
  daemon_quit_(false),
  handle_(nullptr),
  capturing_(false),
  capture_quit_(false),
  queue_(1),
  vid_(-1),
  pid_(-1),
  serial_(serial),
  device_opened_(false)
{
  set_vid_pid(vid_pid);
  if (libusb_init(NULL)) tools::logger()->warn("Unable to init libusb!");

  daemon_thread_ = std::thread{[this] {
    tools::logger()->info("HikRobot's daemon thread started.");

    capture_start();

    while (!daemon_quit_) {
      std::this_thread::sleep_for(100ms);

      if (capturing_) continue;

      capture_stop();
      reset_usb();
      capture_start();
    }

    capture_stop();

    tools::logger()->info("HikRobot's daemon thread stopped.");
  }};
}

HikRobot::~HikRobot()
{
  daemon_quit_ = true;
  if (daemon_thread_.joinable()) daemon_thread_.join();
  tools::logger()->info("HikRobot destructed.");
}

void HikRobot::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  queue_.pop(data);

  img = data.img;
  timestamp = data.timestamp;
}

bool HikRobot::try_read(
  cv::Mat & img, std::chrono::steady_clock::time_point & timestamp, int timeout_ms)
{
  CameraData data;
  if (!queue_.pop(data, std::chrono::milliseconds(timeout_ms))) return false;

  img = data.img;
  timestamp = data.timestamp;
  return true;
}

void HikRobot::capture_start()
{
  capturing_ = false;
  capture_quit_ = false;

  std::lock_guard<std::mutex> sdk_lock(hikrobot_sdk_mutex());

  unsigned int ret;

  MV_CC_DEVICE_INFO_LIST device_list;
  ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_EnumDevices failed: {:#x}", ret);
    return;
  }

  if (device_list.nDeviceNum == 0) {
    tools::logger()->warn("Not found camera!");
    return;
  }

  static std::once_flag logged;
  std::call_once(logged, [&device_list] {
    for (unsigned int i = 0; i < device_list.nDeviceNum; ++i) {
      tools::logger()->info(
        "[HikRobot] device[{}] serial={}", i, usb_serial(device_list.pDeviceInfo[i]));
    }
  });

  MV_CC_DEVICE_INFO * selected = nullptr;
  if (!serial_.empty()) {
    for (unsigned int i = 0; i < device_list.nDeviceNum; ++i) {
      if (usb_serial(device_list.pDeviceInfo[i]) == serial_) {
        selected = device_list.pDeviceInfo[i];
        break;
      }
    }
    if (selected == nullptr) {
      tools::logger()->warn("HikRobot serial \"{}\" not found", serial_);
      return;
    }
  } else {
    selected = device_list.pDeviceInfo[0];
  }

  ret = MV_CC_CreateHandle(&handle_, selected);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_CreateHandle failed: {:#x}", ret);
    handle_ = nullptr;
    return;
  }

  ret = MV_CC_OpenDevice(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_OpenDevice failed: {:#x}", ret);
    MV_CC_DestroyHandle(handle_);
    handle_ = nullptr;
    return;
  }
  device_opened_ = true;
  tools::logger()->info("HikRobot opened, serial={}", usb_serial(selected));

  set_enum_value("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);
  set_enum_value("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
  set_enum_value("GainAuto", MV_GAIN_MODE_OFF);
  set_float_value("ExposureTime", exposure_us_);
  set_float_value("Gain", gain_);
  MV_CC_SetFrameRate(handle_, 150);

  ret = MV_CC_StartGrabbing(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_StartGrabbing failed: {:#x}", ret);
    return;
  }

  capture_thread_ = std::thread{[this] {
    tools::logger()->info("HikRobot's capture thread started.");

    capturing_ = true;

    MV_FRAME_OUT raw;
    MV_CC_PIXEL_CONVERT_PARAM cvt_param;

    while (!capture_quit_) {
      std::this_thread::sleep_for(1ms);

      unsigned int ret;
      unsigned int nMsec = 100;

      ret = MV_CC_GetImageBuffer(handle_, &raw, nMsec);
      if (ret != MV_OK) {
        tools::logger()->warn("MV_CC_GetImageBuffer failed: {:#x}", ret);
        break;
      }

      auto timestamp = std::chrono::steady_clock::now();
      cv::Mat img(cv::Size(raw.stFrameInfo.nWidth, raw.stFrameInfo.nHeight), CV_8U, raw.pBufAddr);

      cvt_param.nWidth = raw.stFrameInfo.nWidth;
      cvt_param.nHeight = raw.stFrameInfo.nHeight;

      cvt_param.pSrcData = raw.pBufAddr;
      cvt_param.nSrcDataLen = raw.stFrameInfo.nFrameLen;
      cvt_param.enSrcPixelType = raw.stFrameInfo.enPixelType;

      cvt_param.pDstBuffer = img.data;
      cvt_param.nDstBufferSize = img.total() * img.elemSize();
      cvt_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed;

      // ret = MV_CC_ConvertPixelType(handle_, &cvt_param);
      const auto & frame_info = raw.stFrameInfo;
      auto pixel_type = frame_info.enPixelType;
      cv::Mat dst_image;
      const static std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes> type_map = {
        {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2RGB},
        {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2RGB},
        {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2RGB},
        {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2RGB}};
      cv::cvtColor(img, dst_image, type_map.at(pixel_type));
      img = dst_image;

      queue_.push({img, timestamp});

      ret = MV_CC_FreeImageBuffer(handle_, &raw);
      if (ret != MV_OK) {
        tools::logger()->warn("MV_CC_FreeImageBuffer failed: {:#x}", ret);
        break;
      }
    }

    capturing_ = false;
    tools::logger()->info("HikRobot's capture thread stopped.");
  }};
}

void HikRobot::capture_stop()
{
  capture_quit_ = true;
  if (capture_thread_.joinable()) capture_thread_.join();

  if (!device_opened_) return;

  std::lock_guard<std::mutex> sdk_lock(hikrobot_sdk_mutex());
  device_opened_ = false;

  unsigned int ret;

  ret = MV_CC_StopGrabbing(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_StopGrabbing failed: {:#x}", ret);
    return;
  }

  ret = MV_CC_CloseDevice(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_CloseDevice failed: {:#x}", ret);
    return;
  }

  ret = MV_CC_DestroyHandle(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_DestroyHandle failed: {:#x}", ret);
    return;
  }
}

void HikRobot::set_float_value(const std::string & name, double value)
{
  unsigned int ret;

  ret = MV_CC_SetFloatValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_SetFloatValue(\"{}\", {}) failed: {:#x}", name, value, ret);
    return;
  }
}

void HikRobot::set_enum_value(const std::string & name, unsigned int value)
{
  unsigned int ret;

  ret = MV_CC_SetEnumValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_SetEnumValue(\"{}\", {}) failed: {:#x}", name, value, ret);
    return;
  }
}

void HikRobot::set_vid_pid(const std::string & vid_pid)
{
  auto index = vid_pid.find(':');
  if (index == std::string::npos) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
    return;
  }

  auto vid_str = vid_pid.substr(0, index);
  auto pid_str = vid_pid.substr(index + 1);

  try {
    vid_ = std::stoi(vid_str, 0, 16);
    pid_ = std::stoi(pid_str, 0, 16);
  } catch (const std::exception &) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
  }
}

void HikRobot::reset_usb() const
{
  if (vid_ == -1 || pid_ == -1) return;

  // 指定了序列号时只复位这一台。同型号海康的 vid:pid 相同，按 vid:pid 复位会打到别的相机。
  if (!serial_.empty()) {
    libusb_device ** devices = nullptr;
    const ssize_t count = libusb_get_device_list(nullptr, &devices);
    if (count < 0) {
      tools::logger()->warn("Unable to list usb devices");
      return;
    }

    for (ssize_t i = 0; i < count; ++i) {
      libusb_device_descriptor desc{};
      if (libusb_get_device_descriptor(devices[i], &desc) != 0) continue;
      if (desc.idVendor != vid_ || desc.idProduct != pid_) continue;

      libusb_device_handle * handle = nullptr;
      if (libusb_open(devices[i], &handle) != 0) continue;

      unsigned char text[256] = {};
      const int length =
        libusb_get_string_descriptor_ascii(handle, desc.iSerialNumber, text, sizeof(text));
      const bool matched = length > 0 && serial_ == reinterpret_cast<char *>(text);
      if (matched) {
        if (libusb_reset_device(handle))
          tools::logger()->warn("Unable to reset usb {}", serial_);
        else
          tools::logger()->info("Reset usb successfully: {}", serial_);
      }
      libusb_close(handle);
      if (matched) break;
    }

    libusb_free_device_list(devices, 1);
    return;
  }

  // https://github.com/ralight/usb-reset/blob/master/usb-reset.c
  auto handle = libusb_open_device_with_vid_pid(NULL, vid_, pid_);
  if (!handle) {
    tools::logger()->warn("Unable to open usb!");
    return;
  }

  if (libusb_reset_device(handle))
    tools::logger()->warn("Unable to reset usb!");
  else
    tools::logger()->info("Reset usb successfully :)");

  libusb_close(handle);
}

}  // namespace io