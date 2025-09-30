#include "PIDController.h"

PIDController::PIDController(double kp,
	double ki,
	double kd,
	double minOutput,
	double maxOutput) noexcept
	: m_kp(kp)
	, m_ki(ki)
	, m_kd(kd)
	, m_minOutput(minOutput)
	, m_maxOutput(maxOutput)
{
}

void PIDController::setGains(double kp, double ki, double kd) noexcept {
	m_kp = kp;
	m_ki = ki;
	m_kd = kd;
}
void PIDController::getGains(double& kp, double& ki, double& kd) const noexcept {
	kp = m_kp;
	ki = m_ki;
	kd = m_kd;
}
void PIDController::setOutputLimits(double minOutput, double maxOutput) noexcept {
	m_minOutput = minOutput;
	m_maxOutput = maxOutput;
}

PIDController::Result PIDController::calculate(double setpoint, double measurement, double dt) noexcept
{
	// 0) dt 안정성 체크
	if (dt < 1e-9) {
		// 이전 출력값을 그대로 반환하거나, P항만 계산하는 등의 처리 가능
		return { m_lastOutput.load(), m_lastError.load() };
	}

	// 1) 오차 계산
	const double error = setpoint - measurement;
	m_lastError.store(error);

	// 2) 비례 항 계산
	const double p_term = m_kp * error;

	// 3) 미분 항 계산 및 1차 필터
	const double rawDeriv = (m_prevMeasurement - measurement) / dt;
	m_derivative = (m_derivativeTau * m_derivative + dt * rawDeriv)
		/ (m_derivativeTau + dt);
	const double d_term = m_kd * m_derivative;

	// 4) 적분 항을 제외한 P + D 합산
	// Integral Windup 방지를 위해 적분 항은 나중에 더함
	const double pd_output = p_term + d_term;

	// 5) 적분 항 계산
	const double i_term = m_ki * (m_integral + error * dt);

	// 6) 출력 한계(clamping) 및 Anti-Windup 적용
	double output = pd_output + i_term;
	double clamped_output = std::clamp(output, m_minOutput, m_maxOutput);

	// 출력이 포화되지 않았을 때만 적분 항을 갱신 (Conditional Integration)
	if (output == clamped_output) {
		m_integral += error * dt;
	}

	// 7) 상태 저장
	m_prevError = error;
	m_prevMeasurement = measurement; // 이전 측정값 업데이트
	m_lastOutput.store(clamped_output);   // 마지막 출력값 업데이트

	return { clamped_output, error };
}

void PIDController::reset() noexcept {
	m_prevError = 0.0;
	m_prevMeasurement = 0.0;
	m_integral = 0.0;
	m_derivative = 0.0;
	m_lastError.store(0.0);
	m_lastOutput.store(0.0);
}
