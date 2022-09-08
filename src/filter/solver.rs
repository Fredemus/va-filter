#[inline]
pub fn tanh_levien_nosimd(x: f64) -> f64 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let a = x + (0.16489087 * x3) + (0.00985468 * x5);
    a / (1.0 + (a * a)).sqrt()
}

/// like rust's asinh, but a way that's less prone to overflow by switching to a formula using smaller values
#[inline]
#[allow(dead_code)]
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

/// solves the nonlinear contributions in the SVF using Newton's method
#[derive(Clone)]
pub(crate) struct DKSolver<const N_N: usize, const N_P: usize, const P_LEN: usize> {
    // current solution of nonlinear contributions
    pub z: [f64; N_N],
    pub last_z: [f64; N_N],
    pub last_p: [f64; N_P],

    // p-vector expanded to the pins of the nonlinear elements
    pub p_full: [f64; P_LEN],


    // temporary storage. Evaluate if necessary
    pub tmp_nn: [f64; N_N],
    pub tmp_np: [f64; N_P],

    // used by the linearization
    factors: [[f64; N_N]; N_N],
    // indices for pivot columns for linearization
    ipiv: [usize; N_N],

    // used by the nonlinear equations

    // full jacobian for the circuit
    pub j: [[f64; N_N]; N_N],
    // full jacobian product for the circuit
    pub jp: [[f64; N_P]; N_N],

    // ?
    pub jq: [[f64; P_LEN]; N_N],
    // the errors of the root-finding for the nonlinear elements
    pub residue: [f64; N_N],
    pub resmaxabs: f64,
}
#[allow(dead_code)]
impl<const N_N: usize, const N_P: usize, const P_LEN: usize> DKSolver<N_N, N_P, P_LEN> {
    pub fn new() -> Self {
        Self {
            z: [0.; N_N],
            last_z: [0.; N_N],
            last_p: [0.; N_P],
            tmp_nn: [0.; N_N],
            tmp_np: [0.; N_P],
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
    // NOTE: this generally works very well but can lead to slow convergence on sudden discontinuities, e.g. the jump in a saw wave
    // In that case maybe a guess from the capacitor states would be better
    pub fn set_extrapolation_origin(
        &mut self,
        p: [f64; N_P],
        z: [f64; N_N],
        // jp: [[f64; N_P]; N_N],
    ) {
        // self.last_jp = jp;
        self.last_p = p;
        self.last_z = z;
    }
    // this entire function could be removed by just using a linearization directly but it would make updating the model(s) require a lot more manual work
    pub fn set_lin_solver(&mut self, new_jacobian: [[f64; N_N]; N_N]) -> bool {
        // const M: usize = N_N;
        // const N: usize = N_N;

        self.factors = new_jacobian;
        // sort of a lower-upper factorization, but storing inverses on the diagonal. Don't remember why
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

    pub fn eval_ota(&self, q: &[f64]) -> (f64, [f64; 2]) {
        let v_in = q[0];
        let i_out = q[1];
        // TODO: switch to tanh approximation
        let tanh_vin = v_in.tanh();
        let residue = tanh_vin + i_out;
        // just a thought: could it be "helped along" by `if jacobian[0] == 0. { jacobian[0] = v_in.signum() * 1e-6}`?
        let jacobian = [(1. - tanh_vin * tanh_vin), 1.0];

        (residue, jacobian)
    }

    pub fn eval_opamp(&self, q: &[f64]) -> (f64, [f64; 2]) {
        let v_in = q[0];
        let v_out = q[1];
        let tanh_vin = tanh_levien_nosimd(v_in);
        // let tanh_vin = v_in.tanh();
        let residue = tanh_vin - v_out;
        // just a thought: could it be "helped along" by `if jacobian[0] == 0. { jacobian[0] = v_in.signum() * 1e-6}`?
        let jacobian = [(1. - tanh_vin * tanh_vin), -1.0];

        (residue, jacobian)
    }

    // testing if the tuple is slow
    pub fn eval_opamp_arr(&self, q: &[f64]) -> [f64; 2] {
        let v_in = q[0];
        let v_out = q[1];
        let tanh_vin = tanh_levien_nosimd(v_in);
        // let tanh_vin = v_in.tanh();
        let residue = tanh_vin - v_out;
        // just a thought: could it be "helped along" by `if jacobian[0] == 0. { jacobian[0] = v_in.signum() * 1e-6}`?
        let mut jacobian = 1. - tanh_vin * tanh_vin;
        if jacobian == 0.0 {
            jacobian = v_in.signum() * 1e-6;
        }
        [residue, jacobian]
    }
    pub fn eval_diode_arr(&self, q: &[f64], i_s: f64, eta: f64) -> [f64; 2] {
        // const V_T_INV: f64 = 1.0 / 25e-3;
        // thermal voltage
        const V_T: f64 = 25e-3;
        // the diode's saturation current. Could make this a function parameter to have slightly mismatched diodes or something
        // const I_S: f64 = 1e-6;
        // const I_S: f64 = 1e-12;
        let v_in = q[0];
        let i_out = q[1];
        let ex = (v_in / (V_T * eta)).exp();
        let residue = i_s * (ex - 1.) - i_out;

        let jacobian = i_s / (V_T * eta) * ex;

        [residue, jacobian]
    }
    // TODO: evaluate if clamping to f32::MAX * 1e-4 or smth would make single-precision solver possible
    pub fn eval_diodepair_arr(&self, q: &[f64], i_s: f64, eta: f64) -> [f64; 2] {
        // the diode's saturation current. Could make this a function parameter to have slightly mismatched diodes or something
        // const I_S: f64 = 1e-6;
        // const I_S: f64 = 1e-12;

        const V_T: f64 = 25e-3;
        let v_t_inv = 1.0 / (V_T * eta);
        let v_in = q[0];
        let i_out = q[1];

        let x = v_in * v_t_inv;
        let ex1 = (x).exp();
        let ex2 = (-x).exp();
        let sinh_vin = i_s * (ex1 - ex2);
        let cosh_vin = i_s * (ex1 + ex2);

        // clamp sinh and cosh since it can go infinite at high drives
        let residue = sinh_vin.clamp(-1e200, 1e200) - i_out;
        let jacobian = cosh_vin.clamp(-1e200, 1e200) * v_t_inv;
        [residue, jacobian]
    }
    // TODO: evaluate if clamping to f32::MAX * 1e-4 or smth would make single-precision solver possible
    pub fn eval_diodepair(&self, q: &[f64], i_s: f64, eta: f64) -> (f64, f64) {
        // the diode's saturation current. Could make this a function parameter to have slightly mismatched diodes or something
        // const I_S: f64 = 1e-6;
        // const I_S: f64 = 1e-12;

        const V_T: f64 = 25e-3;
        let v_t_inv = 1.0 / (V_T * eta);
        let v_in = q[0];
        let i_out = q[1];

        let x = v_in * v_t_inv;
        let ex1 = (x).exp();
        let ex2 = (-x).exp();
        let sinh_vin = i_s * (ex1 - ex2);
        let cosh_vin = i_s * (ex1 + ex2);

        // clamp sinh and cosh since it can go infinite at high drives
        let residue = sinh_vin.clamp(-1e200, 1e200) - i_out;
        let jacobian = cosh_vin.clamp(-1e200, 1e200) * v_t_inv;
        (residue, jacobian)
    }
    // TODO: the diodes end up throwing out absurdly large numbers, need f64 precision or some other way to model diode pairs
    // simple shockley diode equation
    pub fn eval_diode(&self, q: &[f64], i_s: f64, eta: f64) -> (f64, [f64; 2]) {
        // TODO: ideality factor is probably more like ~1.9 than 1
        const ETA: f64 = 1.88;
        // thermal voltage
        // const V_T_INV: f64 = 1.0 / 25e-3;
        const V_T: f64 = 25e-3;
        // the diode's saturation current. Could make this a function parameter to have slightly mismatched diodes or something
        // const I_S: f64 = 1e-6;
        // const I_S: f64 = 1e-12;
        let v_in = q[0];
        let i_out = q[1];
        let ex = (v_in / (V_T * eta)).exp();
        let residue = i_s * (ex - 1.) - i_out;

        let jacobian = [i_s / (V_T * eta) * ex, -1.0];

        (residue, jacobian)
    }
    // inverse of diode clipper to try to avoid having residue/z go to extremely high values.
    // Would likely allow for the solver to be single-precision
    // Sadly has big convergence issues, not sure how to solve them currently,
    pub fn eval_diode_clipper(&self, q: &[f64]) -> (f64, [f64; 2]) {
        // const ETA: f64 = 1.48;
        const ETA: f64 = 1.68;
        const V_T: f64 = 25e-3 * ETA;
        const I_S: f64 = 1e-15;
        const I_S_2: f64 = I_S * I_S;

        let v_in = q[0];
        let i_out = q[1];
        // let residue = V_T * (0.5 * i_out / I_S ).asinh() - v_in;
        let residue = V_T * safe_asinh(0.5 * i_out / I_S) - v_in;

        let jacobian = [-1.0, V_T / (I_S * ((i_out * i_out) / I_S_2 + 4.0).sqrt())];

        (residue, jacobian)
    }
}
