/// solves the nonlinear contributions in the SVF using Newton's method
#[derive(Clone)]
pub(crate) struct DKSolver<const N_N: usize, const N_P: usize, const P_LEN: usize> {
    // current solution of nonlinear contributions
    pub z: [f32; N_N],
    pub last_z: [f32; N_N],
    pub last_p: [f32; N_P],

    // p-vector expanded to the pins of the nonlinear elements
    pub p_full: [f32; P_LEN],

    // last value of jacobian * p
    pub last_jp: [[f32; N_P]; N_N],

    // temporary storage. Evaluate if necessary
    pub tmp_nn: [f32; N_N],
    pub tmp_np: [f32; N_P],

    // used by the linearization
    factors: [[f32; N_N]; N_N],
    // indices for pivot columns for linearization
    ipiv: [usize; N_N],

    // used by the nonlinear equations

    // full jacobian for the circuit
    pub j: [[f32; N_N]; N_N],
    // full jacobian product for the circuit
    pub jp: [[f32; N_P]; N_N],

    // ?
    pub jq: [[f32; P_LEN]; N_N],
    // the errors of the root-finding for the nonlinear elements
    pub residue: [f32; N_N],
    pub resmaxabs: f32,
}
#[allow(dead_code)]
impl<const N_N: usize, const N_P: usize, const P_LEN: usize> DKSolver<N_N, N_P, P_LEN> {
    pub fn new() -> Self {
        Self {
            z: [0.; N_N],
            last_z: [0.; N_N],
            last_p: [0.; N_P],
            last_jp: [[0.; N_P]; N_N],
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

    pub fn set_p(&mut self, p: [f32; N_P], pexps: &[[f32; N_P]; P_LEN]) {
        self.p_full = [0.; P_LEN];
        for i in 0..P_LEN {
            for j in 0..N_P {
                self.p_full[i] += p[j] * pexps[i][j];
            }
        }
    }

    pub fn set_jp(&mut self, pexps: &[[f32; N_P]; P_LEN]) {
        // dbg!(self.jq);
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
        p: [f32; N_P],
        z: [f32; N_N],
        jp: [[f32; N_P]; N_N],
    ) {
        self.last_jp = jp;
        self.last_p = p;
        self.last_z = z;
    }
    // this entire function could be removed by just using a linearization directly but it would make updating the model(s) require a lot more manual work
    pub fn set_lin_solver(&mut self, new_jacobian: [[f32; N_N]; N_N]) {
        // const M: usize = N_N;
        // const N: usize = N_N;

        self.factors = new_jacobian;
        // sort of a lower-upper factorization
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
                // let fkk_inv =  1. / self.factors[k][k];
                self.factors[k][k] = 1. / self.factors[k][k];
                for i in k + 1..N_N {
                    self.factors[i][k] *= self.factors[k][k];
                }
            } else {
                panic!("shouldn't happen");
            }
            // update rest of factors
            for j in k + 1..N_N {
                for i in k + 1..N_N {
                    self.factors[i][j] -= self.factors[i][k] * self.factors[k][j];
                }
            }
        }
    }
    /// based on dgetrs, solve A * X = B with A being lower-upper factorized
    pub fn solve_linear_equation(&self, x: [f32; N_N]) -> [f32; N_N] {
        let mut x_temp = x;
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
        // This loop is wrong somehow
        // TODO: verify this range, should be 3-2-1-0
        for j in (0..N_N).rev() {
            x_temp[j] = self.factors[j][j] * x_temp[j];
            for i in 0..j {
                x_temp[i] -= self.factors[i][j] * x_temp[j];
            }
        }
        x_temp
    }

    pub fn eval_ota(&self, q: &[f32]) -> (f32, [f32; 2]) {
        let v_in = q[0];
        let i_out = q[1];
        // TODO: switch to tanh approximation
        let tanh_vin = v_in.tanh();
        let residue = tanh_vin + i_out;
        // just a thought: could it be "helped along" by `if jacobian[0] == 0. { jacobian[0] = v_in.signum() * 1e-6}`?
        let jacobian = [(1. - tanh_vin * tanh_vin), 1.0];

        (residue, jacobian)
    }

    pub fn eval_opamp(&self, q: &[f32]) -> (f32, [f32; 2]) {
        let v_in = q[0];
        let v_out = q[1];
        // TODO: switch to tanh approximation
        let tanh_vin = v_in.tanh();
        let residue = tanh_vin - v_out;
        // just a thought: could it be "helped along" by `if jacobian[0] == 0. { jacobian[0] = v_in.signum() * 1e-6}`?
        let jacobian = [(1. - tanh_vin * tanh_vin), -1.0];

        (residue, jacobian)
    }
    // simple shockley diode equation
    pub fn eval_diode(&self, q: &[f32]) -> (f32, [f32; 2]) {
        // thermal voltage
        const V_T_INV: f32 = 1.0 / 25e-3;
        // the diode's saturation current. Could make this a function parameter to have slightly mismatched diodes or something
        // const I_S: f32 = 1e-15;
        const I_S: f32 = 1e-12;

        let v_in = q[0];
        let i_out = q[1];
        let ex = (v_in * V_T_INV).exp();

        let residue = I_S * (ex - 1.) - i_out;

        let jacobian = [I_S * V_T_INV * ex, -1.0];

        (residue, jacobian)
    }
}
