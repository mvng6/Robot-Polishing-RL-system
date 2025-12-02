// AirControl.h  
#pragma once  
#include <string>
#include <vector>
#include <atomic>  
#include <chrono>  
#include <algorithm>    // std::clamp  
#include <math.h>
#include "NIDAQmx.h"    // NI-DAQmx  

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// [출력] : PC -> 공압제어 명령값 전달
// [입력] : 공압제어 피드백값 -> PC

/// \brief NI-DAQ를 이용한 Chamber/Spindle 공압 제어기
class AirControl {
public:
	struct Config
	{
		std::string deviceName = "Dev1";        // NI-DAQ 장치 이름
		std::string aoTaskName = "AO_Task";     // AO 태스크 이름
		std::string aiTaskName = "AI_Task";     // AI 태스크 이름

		// 출력 채널 설정 (ao0, ao1 등)
		std::vector<std::string> aoPhysicalChannels = { "ao0", "ao1" }; // Chamber, Spindle
		std::vector<std::string> aiPhysicalChannels = { "ai0", "ai2" }; // Chamber, Spindle 피드백

		// 출력 채널 이름 설정 (자동 생성된 채널명 이름)
		std::vector<std::string> aoChannelNames = { "ChamberChannel", "SpindleChannel" };
		std::vector<std::string> aiChannelNames = { "ChamberFeedback", "SpindleFeedback" };

		// 하드웨어 제약 조건
		double maxVoltage = 10.0;                // 최대 전압 (V 단위)
		double maxChamberPressureMPa = 0.4;      // 최대 Chamber 압력 (MPa 단위)
		double maxSpindlePressureMPa = 0.6;      // 최대 Spindle 압력 (MPa 단위)

		// 하드웨어 상수
		const double MFT_diameter_mm = 94.9;						// MFT 지름 (mm)
		double MFT_radius_m = MFT_diameter_mm / 2.0 * 1e-3;		// MFT 반지름 (m)					
		double A_m2 = M_PI * MFT_radius_m * MFT_radius_m;			// MFT 단면적 (m^2)

		double max_force_N = maxChamberPressureMPa * 1e6 * A_m2;   // 최대 힘 (N 단위) => Chamber 압력에 의한 최대 힘        
		double pid_limit = max_force_N * 1.2;
		const double base_pressure_mpa = 0.2;

		// --- DAQ 샘플링 및 콜백 설정 ---
		uInt64 sampleRateHz = 1000;
		int32 samplesPerCallback = 100; // 몇 샘플만큼 쌓이면 콜백 호출
	};

	// 생성자/소멸자
	explicit AirControl(const Config& config = Config{});
	~AirControl() { releaseTasks(); }

	// getter 메서드 추가
	double getA_m2() const noexcept { return m_config.A_m2; }
	double get_base_pressure_mpa() const noexcept { return m_config.base_pressure_mpa; }

	// 입력 출력 모든 태스크를 초기화하고 시작
	bool initTasks();

	// 입력 출력 모든 태스크를 정지하고 초기화 상태로 되돌림
	void releaseTasks() noexcept;

	// 모든 태스크가 초기화되었는지 여부 조회
	bool isInitialized() const noexcept;

	// [출력] 현재 목표 압력을 읽어서 전압값 계산후 업데이트
	void updateOutputs() noexcept;

	// Chamber 목표 압력 설정 시 하드웨어 범위로 제한하는 값까지 클램핑
	void setDesiredChamberPressure(double pMPa) noexcept {
		// 소수점 3자리로 반올림하여 명확한 제어 정밀도 보장
		double rounded = std::round(pMPa * 1000.0) / 1000.0;
		m_desiredChamber = std::clamp(rounded, 0.0, m_config.maxChamberPressureMPa);
	}
	// Chamber 압력 조회
	double desiredChamberPressure() const noexcept { return m_desiredChamber.load(); }

	// Spindle 목표 압력 설정 시 하드웨어 범위로 제한하는 값까지 클램핑
	void setDesiredSpindlePressure(double pMPa) noexcept {
		// 소수점 3자리로 반올림하여 명확한 제어 정밀도 보장
		double rounded = std::round(pMPa * 1000.0) / 1000.0;
		m_desiredSpindle = std::clamp(rounded, 0.0, m_config.maxSpindlePressureMPa);
	}
	// Spindle 압력 조회
	double desiredSpindlePressure() const noexcept { return m_desiredSpindle.load(); }

	// [출력] Chamber & Spindle 송신 값 전압 조회
	double sendChamberVoltage() const noexcept { return m_sendVoltChamber.load(); }
	double sendSpindleVoltage() const noexcept { return m_sendVoltSpindle.load(); }
	double sendChamberPressure() const noexcept { return m_sendPressChamber.load(); }
	double sendSpindlePressure() const noexcept { return m_sendPressSpindle.load(); }

	// [입력] Chamber & Spindle 피드백 전압 및 압력 조회
	double feedbackChamberVoltage() const noexcept { return m_feedbackVoltChamber.load(); }
	double feedbackSpindleVoltage() const noexcept { return m_feedbackVoltSpindle.load(); }
	double feedbackChamberPressure() const noexcept { return m_feedbackPressChamber.load(); }
	double feedbackSpindlePressure() const noexcept { return m_feedbackPressSpindle.load(); }


private:
	// AO 및 AI 태스크를 생성하고 채널을 추가하는 내부 함수
	bool setupAoTask();
	bool setupAiTask();
	bool checkDaqError(int32 error, const char* context) noexcept;
	static int32 CVICALLBACK EveryNCallback(TaskHandle taskHandle, int32 everyNsamplesEventType, uInt32 nSamples, void* callbackData);

	Config m_config;

	TaskHandle m_aoTask{ nullptr };                                                     // 출력(AO) 태스크 핸들 (명령)
	TaskHandle m_aiTask{ nullptr };                                                     // 입력(AI) 태스크 핸들 (피드백)

	// --- 통계 ---
	std::atomic<long long> m_totalSamplesRead{ 0 };                                     // 총 샘플 읽은 수
	std::chrono::steady_clock::time_point m_startTime;                                  // 태스크 시작 시간
	std::atomic<double> m_desiredChamber{ 0.0 }, m_desiredSpindle{ 0.0 };               // 목표 목표치 (MPa 단위)

	// 출력(AO) 값 ---
	std::atomic<double> m_sendVoltChamber{ 0.0 }, m_sendVoltSpindle{ 0.0 };             // 송신 전압값 (V 단위)
	std::atomic<double> m_sendPressChamber{ 0.0 }, m_sendPressSpindle{ 0.0 };           // 송신 압력값 (MPa 단위)

	// --- 입력(AI) 값 ---
	std::atomic<double> m_feedbackVoltChamber{ 0.0 }, m_feedbackVoltSpindle{ 0.0 };     // 피드백 전압값 (V 단위)
	std::atomic<double> m_feedbackPressChamber{ 0.0 }, m_feedbackPressSpindle{ 0.0 };   // 피드백 압력값 (MPa 단위)
};
