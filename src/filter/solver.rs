#[inline(always)]
pub fn tanh_levien(x: f64) -> f64 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let a = x + (0.16489087 * x3) + (0.00985468 * x5);
    a / (1.0 + (a * a)).sqrt()
}

/// like rust's asinh, but a way that's less prone to overflow by switching to a formula using smaller values
#[inline]
pub fn safe_asinh(x: f64) -> f64 {
    const AH_LN2: f64 = 6.93147180559945286227e-01;
    let x_abs = x.abs();
    let w = if x_abs < 2.0_f64.powi(28) {
        (x_abs + ((x * x) + 1.0).sqrt()).ln()
    }
    // if x is very high, use simpler formula (x * x in the formula above can make asinh go inf)
    else {
        x_abs.ln() + AH_LN2
    };
    w.copysign(x)
}

/// Provides the data structures and methods to solve the nonlinear contributions in the filter models using Newton's method
/// Requires the filter model to implement a method for evaluating the nonlinearities themself
/// since I didn't want to deal with passing functions into the new() method
#[derive(Clone)]
pub struct DKSolver<const N_N: usize, const N_P: usize, const P_LEN: usize> {
    // current solution of nonlinear contributions
    pub z: [f64; N_N],
    pub last_z: [f64; N_N],
    pub last_p: [f64; N_P],

    // p-vector expanded to the pins of the nonlinear elements
    pub p_full: [f64; P_LEN],

    // used by the linearization. These 2 fields stores a lower-upper factorization of j
    factors: [[f64; N_N]; N_N],
    // indices for pivot columns for linearization
    ipiv: [usize; N_N],

    // used by the nonlinear equations

    // full jacobian for the circuit
    pub j: [[f64; N_N]; N_N],
    // The jacobian as it applies to the p-vector (voltage/current into nonlinear elements) for the circuit
    pub jp: [[f64; N_P]; N_N],

    // the jacobian as it applies to q, that is the p-vector and how the z-vector (voltage/current equivalent of the nonlinearities)
    pub jq: [[f64; P_LEN]; N_N],
    // the errors of the root-finding for the nonlinear elements
    pub residue: [f64; N_N],
    pub resmaxabs: f64,
}
impl<const N_N: usize, const N_P: usize, const P_LEN: usize> DKSolver<N_N, N_P, P_LEN> {
    pub fn new() -> Self {
        Self {
            z: [0.; N_N],
            last_z: [0.; N_N],
            last_p: [0.; N_P],
            factors: [[0.; N_N]; N_N],
            ipiv: [0; N_N],
            j: [[0.; N_N]; N_N],
            jp: [[0.; N_P]; N_N],
            p_full: [0.; P_LEN],
            jq: [[0.; P_LEN]; N_N],
            residue: [0.; N_N],
            resmaxabs: 0.,
        }
    }

    pub fn set_p(&mut self, p: [f64; N_P], pexps: &[[f64; N_P]; P_LEN]) {
        self.p_full = [0.; P_LEN];
        for i in 0..P_LEN {
            for j in 0..N_P {
                self.p_full[i] += p[j] * pexps[i][j];
            }
        }
    }

    pub fn set_jp(&mut self, pexps: &[[f64; N_P]; P_LEN]) {
        for i in 0..N_N {
            for j in 0..N_P {
                self.jp[i][j] = 0.;
                for k in 0..P_LEN {
                    self.jp[i][j] += self.jq[i][k] * pexps[k][j];
                }
            }
        }
    }

    // prepare the solver for next sample by storing the current solution, so it can be used as an initial guess
    pub fn set_extrapolation_origin(&mut self, p: [f64; N_P], z: [f64; N_N]) {
        self.last_p = p;
        self.last_z = z;
    }

    pub fn set_lin_solver(&mut self, new_jacobian: [[f64; N_N]; N_N]) -> bool {
        self.factors = new_jacobian;
        for k in 0..N_N {
            let mut kp = k;
            let mut amax = 0.0;
            for i in k..N_N {
                let absi = self.factors[i][k].abs();
                if absi > amax {
                    kp = i;
                    amax = absi;
                }
            }
            self.ipiv[k] = kp;
            if self.factors[kp][k] != 0.0 {
                if k != kp {
                    // interchange values
                    for i in 0..N_N {
                        let tmp = self.factors[k][i];
                        self.factors[k][i] = self.factors[kp][i];
                        self.factors[kp][i] = tmp;
                    }
                }
                // scale first column
                self.factors[k][k] = 1. / self.factors[k][k];
                for i in k + 1..N_N {
                    self.factors[i][k] *= self.factors[k][k];
                }
            } else {
                // Jacobian for a nonlin element contains a 0 value, making it impossible to iterate with
                return false;
            }
            // update rest of factors
            for j in k + 1..N_N {
                for i in k + 1..N_N {
                    self.factors[i][j] -= self.factors[i][k] * self.factors[k][j];
                }
            }
        }
        true
    }
    /// based on dgetrs, solve A * X = B with A being lower-upper factorized
    pub fn solve_linear_equations(&self, b: [f64; N_N]) -> [f64; N_N] {
        let mut x_temp = b;
        for i in 0..N_N {
            // x[i], x[self.ipiv[i]] =
            x_temp.swap(i, self.ipiv[i]);
        }
        for j in 0..N_N {
            let xj = x_temp[j];
            for i in j + 1..N_N {
                x_temp[i] -= self.factors[i][j] * xj;
            }
        }
        for j in (0..N_N).rev() {
            x_temp[j] = self.factors[j][j] * x_temp[j];
            for i in 0..j {
                x_temp[i] -= self.factors[i][j] * x_temp[j];
            }
        }
        x_temp
    }

    #[inline(always)]
    pub fn eval_opamp(&self, v_in: f64, v_out: f64) -> (f64, [f64; 2]) {
        let tanh_vin = tanh_levien(v_in);
        let residue = tanh_vin - v_out;
        let mut jacobian = [(1. - tanh_vin * tanh_vin), -1.0];
        // this if-statement sort of helps the solver converge, since 0-valued entries in the jacobian can't be used for iteration
        if jacobian[0] == 0.0 {
            jacobian[0] = v_in.signum() * 1e-9;
        }
        (residue, jacobian)
    }
    #[inline]
    pub fn eval_ota(&self, q: &[f64]) -> (f64, [f64; 2]) {
        let v_in = q[0];
        let i_out = q[1];
        let tanh_vin = tanh_levien(v_in);
        let residue = tanh_vin + i_out;
        let mut jacobian = [(1. - tanh_vin * tanh_vin), 1.0];
        if jacobian[0] == 0.0 {
            jacobian[0] = v_in.signum() * 1e-9;
        }
        (residue, jacobian)
    }

    // TODO: evaluate if clamping to f32::MAX * 1e-4 or smth would make single-precision solver possible
    // looks like the svf at least gets much worse convergence when using a single-precision solver
    /// 2 shockley diodes, with the + pin of one connected to the other's - pin and vice versa
    #[inline(always)]
    pub fn eval_diodepair(&self, v_in: f64, i_out: f64, i_s: f64, eta: f64) -> (f64, [f64; 2]) {
        // the diode's saturation current. Could make this a function parameter to have slightly mismatched diodes or something
        // const I_S: f64 = 1e-6;
        // const I_S: f64 = 1e-12;

        const V_T: f64 = 25e-3;
        let v_t_inv = 1.0 / (V_T * eta);

        let x = v_in * v_t_inv;
        let ex1 = (x).exp();
        let ex2 = (-x).exp();
        let sinh_vin = i_s * (ex1 - ex2);
        let cosh_vin = i_s * (ex1 + ex2);

        // clamp sinh and cosh since they can go infinite at high drives
        const LIM: f64 = 1e34;
        let residue = sinh_vin.clamp(-LIM, LIM) - i_out;
        let jacobian = cosh_vin.clamp(-LIM, LIM) * v_t_inv;
        (residue, [jacobian, -1.])
    }
    // simple shockley diode equation
    pub fn eval_diode(&self, q: &[f64], i_s: f64, eta: f64) -> (f64, [f64; 2]) {
        // thermal voltage
        const V_T: f64 = 25e-3;

        let v_in = q[0];
        let i_out = q[1];
        let ex = (v_in / (V_T * eta)).exp();
        let residue = i_s * (ex - 1.) - i_out;

        let jacobian = [i_s / (V_T * eta) * ex, -1.0];

        (residue, jacobian)
    }
    // inverse of diode clipper to try to avoid having residue/z go to extremely high values.
    // sadly has big convergence issues, due to the extremely steep slope at 0, not sure how to solve them currently,
    pub fn eval_diode_clipper(&self, q: &[f64]) -> (f64, [f64; 2]) {
        const ETA: f64 = 1.68;
        const V_T: f64 = 25e-3 * ETA;
        const I_S: f64 = 1e-15;
        const I_S_2: f64 = I_S * I_S;

        let v_in = q[0];
        let i_out = q[1];
        let residue = V_T * safe_asinh(0.5 * i_out / I_S) - v_in;

        let jacobian = [-1.0, V_T / (I_S * ((i_out * i_out) / I_S_2 + 4.0).sqrt())];

        (residue, jacobian)
    }
}
