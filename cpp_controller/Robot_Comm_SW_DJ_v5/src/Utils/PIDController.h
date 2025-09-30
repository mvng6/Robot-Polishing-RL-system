#pragma once

#include <algorithm>    // std::clamp
#include <atomic>


class PIDController {
public:
	// 결과 구조체 정의
	struct Result {
		double output;
		double error;
	};

	// 생성자
	PIDController(
		double kp = 0.0,                                                        // PID 비례 게인 초기화
		double ki = 0.0,    										            // PID 적분 게인 초기화
		double kd = 0.0,    						                            // PID 미분 게인 초기화
		double minOutput = -1.0,                                                // PID 출력 최소 한계 초기화
		double maxOutput = +1.0) noexcept;                                      // PID 출력 최대 한계 초기화

	void setGains(double kp, double ki, double kd) noexcept;                    // PID 게인 설정 함수
	void getGains(double& kp, double& ki, double& kd) const noexcept;			// PID 게인 읽는 함수
	void setOutputLimits(double minOutput, double maxOutput) noexcept;          // PID 출력 한계 설정 함수

	Result calculate(                                                           // PID 계산 함수                                                       
		double setpoint,                                                        // 목표값
		double measurement,                                                     // 측정값
		double dt) noexcept;                    				                // 시간 간격 (초 단위)

	void setIntegral(double integralValue) {
		this->m_integral = integralValue;
	}

	void reset() noexcept;                                                      // PID 상태 초기화

	double getKp()        const noexcept { return m_kp; }                       // PID 비례 게인 읽는 함수
	double getKi()        const noexcept { return m_ki; }                       // PID 적분 게인 읽는 함수
	double getKd()        const noexcept { return m_kd; }						// PID 미분 게인 읽는 함수
	double getLastError() const noexcept { return m_lastError.load(); }			// 마지막 오차 읽는 함수

private:

	double m_kp;																// PID 비례 게인
	double m_ki;																// PID 적분 게인
	double m_kd;																// PID 미분 게인

	double m_minOutput;															// PID 출력 최소 한계
	double m_maxOutput;															// PID 출력 최대 한계

	double m_prevError{ 0.0 };													// 이전 오차 (미분 계산용)	
	double m_prevMeasurement{ 0.0 };											// 이전 측정값 (미분 계산용)

	double m_integral{ 0.0 };													// 적분 항 (오차의 누적 합산)
	double m_derivative{ 0.0 };													// 미분 항 (오차 변화율)	

	double m_derivativeTau{ 0.01 };												// 미분 필터 타우 (1차 필터링을 위한 시간 상수, 초 단위)

	std::atomic<double> m_lastOutput{ 0.0 };													// 마지막 출력값 (이전 PID 출력값 저장용)
	std::atomic<double> m_lastError{ 0.0 };										// 마지막 오차 (원자적 연산을 위한 atomic 변수)
};
