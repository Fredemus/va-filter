use crate::{
    filter::DKSolver,
    filter_params::{FilterParams, SvfMode},
    utils::AtomicOps,
};
use std::f32::consts::{FRAC_1_SQRT_2, SQRT_2};
use std::f64::consts::{FRAC_1_SQRT_2 as FRAC_1_SQRT_2_F64, SQRT_2 as SQRT_2_F64};
use std::sync::Arc;

pub struct Svf {
    filters: [SvfCoreFast; 2],
}

impl Svf {
    pub fn new(params: Arc<FilterParams>) -> Self {
        Self {
            filters: [SvfCoreFast::new(params.clone()), SvfCoreFast::new(params)],
        }
    }
    pub fn process(&mut self, input: [f32; 2]) -> [f32; 2] {
        [
            self.filters[0].tick_dk(input[0]),
            self.filters[1].tick_dk(input[1]),
        ]
    }
    pub fn update(&mut self) {
        self.filters[0].update_matrices();
        self.filters[1].update_matrices();
    }
    pub fn reset(&mut self) {
        self.filters[0].reset();
        self.filters[1].reset();
    }
}

const N_P: usize = 3;
const N_N: usize = 3;
const P_LEN: usize = 6;
const N_OUTS: usize = 3;
const N_STATES: usize = 2;
const TOL: f64 = 1e-5;
/// 2-pole state-variable filter
pub struct SvfCore {
    pub params: Arc<FilterParams>,
    pub vout: [f32; N_OUTS],
    pub s: [f32; N_STATES],

    // used to find the nonlinear contributions
    dq: [[f32; N_STATES]; N_P],
    eq: [f32; N_P],
    pub fq: [[f64; N_N]; P_LEN],
    // dq, eq are actually much larger, pexps are used to reduce them
    pexps: [[f64; N_P]; P_LEN],

    // used to update the capacitor states
    a: [[f32; N_STATES]; N_STATES],
    b: [f32; N_STATES],
    c: [[f32; N_N]; N_STATES],

    // used to find the output values
    // dy: [[f32; 2]; 2],
    dy: [[f32; N_STATES]; N_OUTS],
    ey: [f32; N_OUTS],
    fy: [[f32; N_N]; N_OUTS],

    solver: DKSolver<N_N, N_P, P_LEN>,
}

impl SvfCore {
    pub fn new(params: Arc<FilterParams>) -> Self {
        // TODO: pass in proper params
        let fs = params.sample_rate.get();
        let g = (std::f32::consts::PI * 1000. / (fs as f32)).tan();
        let res = 0.1 - 1.;
        let g_f64 = g as f64;
        let res_f64 = res as f64;

        let pexps = [
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
            [SQRT_2_F64 * g_f64, 0., 0.],
            [0., 0., 1.],
        ];
        let fq = [
            [0., 1., 0.],
            [-FRAC_1_SQRT_2_F64 / (2. * g_f64), 0., 0.],
            [-FRAC_1_SQRT_2_F64 * g_f64, 0., -1.],
            [0., 0., 1.],
            [0.5, -1., 0.],
            [-0.5 * res_f64 - FRAC_1_SQRT_2_F64 / (2. * g_f64), 3., 1.],
        ];
        let mut a = Self {
            params,
            vout: [0.; N_OUTS],
            s: [0.; 2],

            dq: [
                [FRAC_1_SQRT_2 / (2. * g), 0.],
                [-FRAC_1_SQRT_2 * g, -1.],
                [FRAC_1_SQRT_2 / (2. * g) - 0.5 * res, 0.],
            ],
            eq: [0., 0., 1.],
            fq,
            pexps,

            a: [[0., 0.], [SQRT_2 * g, 1.]],
            b: [0., 0.],
            c: [[1., 0., 0.], [SQRT_2 * g, 0., 0.]],

            dy: [[0., 0.], [-0.5, 0.], [FRAC_1_SQRT_2 / (2. * g), 0.]],
            ey: [0., 0., 0.],
            fy: [
                [0., 0., 1.],
                [-0.5, 0., 0.],
                [-FRAC_1_SQRT_2 / (2. * g), 0., 0.],
            ],

            solver: DKSolver::new(),
        };
        a.solver.set_p([0.; N_P], &pexps);
        a.evaluate_nonlinearities([0.; N_N], fq);
        a.solver.set_lin_solver(a.solver.j);
        a.solver.set_jp(&pexps);
        a.solver.set_extrapolation_origin([0.; N_P], [0.; N_N]);

        a
    }
    pub fn reset(&mut self) {
        self.s = [0.; 2];
        self.solver.set_p([0.; N_P], &self.pexps);
        self.evaluate_nonlinearities([0.; N_N], self.fq);
        self.solver.set_lin_solver(self.solver.j);
        self.solver.set_jp(&self.pexps);
        self.solver.set_extrapolation_origin([0.; N_P], [0.; N_N]);
    }
    pub fn update_matrices(&mut self) {
        // let fs = self.params.sample_rate.get();

        let g = self.params.g.get() * 2.;
        let res = self.params.zeta.get() - 1.0;
        let g_f64 = g as f64;
        let res_f64 = res as f64;

        self.fq = [
            [0., 1., 0.],
            [-FRAC_1_SQRT_2_F64 / (2. * g_f64), 0., 0.],
            [-FRAC_1_SQRT_2_F64 * g_f64, 0., -1.],
            [0., 0., 1.],
            [0.5, -1., 0.],
            [-0.5 * res_f64 - FRAC_1_SQRT_2_F64 / (2. * g_f64), 3., 1.],
        ];
        self.pexps[4][0] = SQRT_2_F64 * g_f64;

        self.dy[2][0] = FRAC_1_SQRT_2 / (2. * g);
        self.fy[2][0] = -FRAC_1_SQRT_2 / (2. * g);

        self.dq[0][0] = FRAC_1_SQRT_2 / (2. * g);
        self.dq[1][0] = -FRAC_1_SQRT_2 * g;
        self.dq[2][0] = FRAC_1_SQRT_2 / (2. * g) - 0.5 * res;

        self.c[1][0] = SQRT_2 * g;
        self.a[1][0] = SQRT_2 * g;
    }
    // TODO: pls roll the updates into for loops
    pub fn tick_dk(&mut self, input: f32) -> f32 {
        // -input since the svf inverts it
        let input = -input * (self.params.drive.value());

        // let p = dot(dq, s) + dot(eq, input);
        let mut p = [0.; N_P];
        // find the
        p[0] = (self.dq[0][0] * self.s[0] + self.dq[0][1] * self.s[1] + self.eq[0] * input) as f64;
        p[1] = (self.dq[1][0] * self.s[0] + self.dq[1][1] * self.s[1] + self.eq[1] * input) as f64;
        p[2] = (self.dq[2][0] * self.s[0] + self.dq[2][1] * self.s[1] + self.eq[2] * input) as f64;

        //
        // find nonlinear contributions (solver.z), applying homotopy if it fails to converge
        self.homotopy_solver(p);
        // self.nonlinear_contribs(p);

        self.vout[0] = self.dy[0][0] * self.s[0] + self.dy[0][1] * self.s[1] + self.ey[0] * input;
        self.vout[1] = self.dy[1][0] * self.s[0] + self.dy[1][1] * self.s[1] + self.ey[1] * input;
        self.vout[2] = self.dy[2][0] * self.s[0] + self.dy[2][1] * self.s[1] + self.ey[2] * input;

        for i in 0..N_OUTS {
            for j in 0..N_N {
                self.vout[i] += self.solver.z[j] as f32 * self.fy[i][j];
            }
        }
        let s1_update = self.a[1][0] * self.s[0] + self.a[1][1] * self.s[1];
        self.s[0] = self.a[0][0] * self.s[0] + self.a[0][1] * self.s[1] + self.b[0] * input;
        self.s[1] = s1_update + self.b[1] * input;
        for i in 0..N_STATES {
            for j in 0..N_N {
                self.s[i] += self.solver.z[j] as f32 * self.c[i][j];
            }
        }

        self.get_output(input, self.params.zeta.get())
    }

    pub fn homotopy_solver(&mut self, p: [f64; N_P]) {
        self.nonlinear_contribs(p);
        // if the newton solver failed to converge, apply homotopy
        if self.solver.resmaxabs >= TOL {
            let mut a = 0.5;
            let mut best_a = 0.;
            while best_a < 1. {
                let mut pa = self.solver.last_p;

                for i in 0..pa.len() {
                    pa[i] = pa[i] * (1. - a);
                    pa[i] = pa[i] + a * p[i];
                }
                self.nonlinear_contribs(pa);
                if self.solver.resmaxabs < TOL {
                    best_a = a;
                    a = 1.0;
                } else {
                    let new_a = (a + best_a) / 2.;
                    if !(best_a < new_a && new_a < a) {
                        // no values between a and best_a. This means the homotopy failed to find an in-between value for the solution
                        break;
                    }
                    a = new_a;
                }
            }
            // if self.solver.resmaxabs >= TOL {
            //     println!("failed to converge. residue: {:?}", self.solver.residue);
            //     for arr in &self.solver.j {
            //         for x in arr {
            //             if !x.is_finite() {
            //                 panic!("solution contains infinite/NaN value");
            //             }
            //         }
            //     }
            // }
        }
    }

    // uses newton's method to find the nonlinear contributions in the circuit. Not guaranteed to converge
    fn nonlinear_contribs(&mut self, p: [f64; N_P]) {
        self.solver.set_p(p, &self.pexps);

        let mut tmp_np = [0.; N_P];
        tmp_np[0] = p[0] - self.solver.last_p[0];
        tmp_np[1] = p[1] - self.solver.last_p[1];
        tmp_np[2] = p[2] - self.solver.last_p[2];

        let mut tmp_nn = [0.; N_N];
        for i in 0..N_N {
            tmp_nn[i] = 0.;
            for j in 0..N_P {
                tmp_nn[i] += self.solver.jp[i][j] * tmp_np[j];
            }
        }
        tmp_nn = self.solver.solve_linear_equations(tmp_nn);

        // self.solver.z = self.solver.last_z - self.solver.tmp_nn;
        for i in 0..self.solver.z.len() {
            self.solver.z[i] = self.solver.last_z[i] - tmp_nn[i];
        }
        for _plsconverge in 0..100 {
            self.evaluate_nonlinearities(self.solver.z, self.fq);

            self.solver.resmaxabs = 0.;
            for x in &self.solver.residue {
                if x.is_finite() {
                    if x.abs() > self.solver.resmaxabs {
                        self.solver.resmaxabs = x.abs();
                    }
                } else {
                    // if any of the residue have become NaN/inf, stop early.
                    // If using the homotopy solver, it will kick in and find an alternate, slower path to convergence
                    self.solver.resmaxabs = 1000.;
                    return;
                }
            }

            self.solver.set_lin_solver(self.solver.j);
            if self.solver.resmaxabs < TOL {
                // dbg!(_plsconverge);
                break;
            }

            // update z with the linsolver according to the residue
            tmp_nn = self.solver.solve_linear_equations(self.solver.residue);

            for i in 0..self.solver.z.len() {
                self.solver.z[i] -= tmp_nn[i];
            }
        }
        if self.solver.resmaxabs < TOL {
            self.solver.set_jp(&self.pexps);
            self.solver.set_extrapolation_origin(p, self.solver.z);
        }
        // else {
        // panic!("failed to converge. residue: {:?}", self.solver.residue);
        // println!("failed to converge. residue: {:?}", self.solver.residue);
        // }
    }

    fn evaluate_nonlinearities(&mut self, z: [f64; N_N], fq: [[f64; N_N]; P_LEN]) {
        // TODO: better way of finding dot-product between fq and z
        let mut dot_p = [0.; P_LEN];
        let mut q = self.solver.p_full;
        for i in 0..P_LEN {
            for j in 0..N_N {
                dot_p[i] += z[j] * fq[i][j];
            }
            q[i] += dot_p[i];
        }
        let (res1, jq1) = self.solver.eval_opamp(q[0], q[1]);
        let (res2, jq2) = self.solver.eval_opamp(q[2], q[3]);

        let (res3, jq3) = self.solver.eval_diodepair(q[4], q[5], 1e-12, 1.28);

        // TODO: consider simplifying jq
        self.solver.jq[0][0] = jq1[0];
        self.solver.jq[0][1] = jq1[1];

        self.solver.jq[1][2] = jq2[0];
        self.solver.jq[1][3] = jq2[1];

        self.solver.jq[2][4] = jq3[0];
        self.solver.jq[2][5] = jq3[1];

        // update j to the matrix product fq * jq
        for i in 0..self.solver.jq.len() {
            for j in 0..N_N {
                self.solver.j[i][j] = 0.;
                for k in 0..P_LEN {
                    self.solver.j[i][j] += self.solver.jq[i][k] * fq[k][j];
                }
            }
        }
        self.solver.residue = [res1, res2, res3];
    }
    // highpass and notch doesn't work right, likely because `input` isn't quite defined right. Prolly doesn't need to be subtracted?
    // ^ seems to be fixed now?
    fn get_output(&self, input: f32, k: f32) -> f32 {
        match self.params.mode.value() {
            SvfMode::LP => self.vout[0],  // lowpass
            SvfMode::HP => self.vout[2],  // highpass
            SvfMode::BP1 => self.vout[1], // bandpass
            // the notch isn't limited to the -1 to 1 range like the other modes, not sure how to solve nicely for it currently
            SvfMode::Notch => input + k * self.vout[1], // notch
            //3 => input + 2. * k * self.vout[1], // allpass
            SvfMode::BP2 => k * self.vout[1], // bandpass (normalized peak gain)
                                              // _ => input + 2. * self.vout[1] + k * self.vout[0], // peak / resonator thingy
        }
    }
}

pub struct SvfCoreFast {
    pub params: Arc<FilterParams>,
    pub vout: [f32; N_OUTS],
    pub s: [f32; N_STATES],

    pub fq: [[f64; N_N]; P_LEN],
    c1: f32,
    c2: f32,
    c3: f32,

    solver: DKSolver<N_N, N_P, P_LEN>,
    // for storing the jacobian for the q (p + dot(z, fq) vector
    jq: [f64; 6],
}

impl SvfCoreFast {
    pub fn new(params: Arc<FilterParams>) -> Self {
        let fs = params.sample_rate.get();
        let g = (std::f32::consts::PI * 1000. / (fs as f32)).tan();
        let res = 0.1;
        let g_f64 = g as f64;
        let res_f64 = res as f64;

        let fq = [
            [0., 1., 0.],
            [-FRAC_1_SQRT_2_F64 / (2. * g_f64), 0., 0.],
            [-FRAC_1_SQRT_2_F64 * g_f64, 0., -1.],
            [0., 0., 1.],
            [0.5, -1., 0.],
            [-0.5 * res_f64 - FRAC_1_SQRT_2_F64 / (2. * g_f64), 3., 1.],
        ];
        let mut a = Self {
            params,
            vout: [0.; N_OUTS],
            s: [0.; 2],

            fq,
            c1: FRAC_1_SQRT_2 / (2. * g),
            c2: SQRT_2 * g,
            c3: FRAC_1_SQRT_2 / (2. * g) - 0.5 * res,
            jq: [0., -1., 0., -1., 0., 1.],

            solver: DKSolver::new(),
        };
        a.reset();
        a
    }
    pub fn reset(&mut self) {
        self.s = [0.; 2];
        self.evaluate_nonlinearities([0.; N_N], self.fq);
        self.solver.set_lin_solver(self.solver.j);
        self.solver.set_extrapolation_origin([0.; N_P], [0.; N_N]);
    }
    pub fn update_matrices(&mut self) {
        let g = self.params.g.get()  * 2.;
        let res = self.params.zeta.get();
        let g_f64 = g as f64;
        let res_f64 = res as f64;

        self.fq = [
            [0., 1., 0.],
            [-FRAC_1_SQRT_2_F64 / (2. * g_f64), 0., 0.],
            [-FRAC_1_SQRT_2_F64 * g_f64, 0., -1.],
            [0., 0., 1.],
            [0.5, -1., 0.],
            [-0.5 * res_f64 - FRAC_1_SQRT_2_F64 / (2. * g_f64), 3., 1.],
        ];
        self.c1 = FRAC_1_SQRT_2 / (2. * g);
        self.c2 = SQRT_2 * g;
        self.c3 = FRAC_1_SQRT_2 / (2. * g) - 0.5 * res;
    }
    pub fn tick_dk(&mut self, input: f32) -> f32 {
        // - input since the svf inverts the signal
        let input = (-input * self.params.drive.value()).clamp(-10., 10.);

        let mut p = [0.; N_P];
        p[0] = (self.c1 * self.s[0]) as f64;
        p[1] = (-0.5 * self.c2 * self.s[0] - self.s[1]) as f64;
        p[2] = (self.c3 * self.s[0] + input) as f64;

        // find nonlinear contributions (solver.z), applying homotopy if it fails to converge
        self.homotopy_solver(p);

        self.vout[0] = self.solver.z[2] as f32;
        self.vout[1] = -0.5 * self.s[0] - 0.5 * self.solver.z[0] as f32;
        self.vout[2] = self.c1 * self.s[0] - self.c1 * self.solver.z[0] as f32;

        self.s[1] = self.c2 * self.s[0] + self.s[1] + self.c2 * self.solver.z[0] as f32;
        self.s[0] = self.solver.z[0] as f32;

        self.get_output(input, self.params.zeta.get())
    }

    pub fn homotopy_solver(&mut self, p: [f64; N_P]) {
        self.nonlinear_contribs(p);
        // if the newton solver failed to converge, apply homotopy
        if self.solver.resmaxabs >= TOL {
            // println!("needs homotopy. p: {:?}, start z: {:?}", p, self.solver.z);
            let mut a = 0.5;
            let mut best_a = 0.;
            while best_a < 1. {
                let mut pa = self.solver.last_p;

                for i in 0..pa.len() {
                    pa[i] = pa[i] * (1. - a);
                    pa[i] = pa[i] + a * p[i];
                }
                self.nonlinear_contribs(pa);
                if self.solver.resmaxabs < TOL {
                    best_a = a;
                    a = 1.0;
                } else {
                    let new_a = (a + best_a) / 2.;
                    if !(best_a < new_a && new_a < a) {
                        // no values between a and best_a. This could probably be simplified
                        break;
                    }
                    a = new_a;
                }
            }
            // this doesn't seem to ever happen anymore
            // if self.solver.resmaxabs >= TOL {
            //     println!("failed to converge. residue: {:?}", self.solver.residue);
            //     // println!("z: {:?}", self.solver.z);
            //     for x in &self.solver.z {
            //         if !x.is_finite() || x.abs() > 1e100 {
            //             panic!("solution contains infinite/NaN/ extremely large value");
            //         }
            //     }
            //     for arr in &self.solver.j {
            //         for x in arr {
            //             if !x.is_finite() {
            //                 panic!("solution contains infinite/NaN value");
            //             }
            //         }
            //     }
            // }
        }
    }

    // uses newton's method to find the nonlinear contributions in the circuit. Not guaranteed to converge
    fn nonlinear_contribs(&mut self, p: [f64; N_P]) {
        self.solver.p_full[1] = p[0];
        self.solver.p_full[2] = p[1];
        self.solver.p_full[4] = self.c2 as f64 * p[0];
        self.solver.p_full[5] = p[2];

        let tmp_np = [
            p[0] - self.solver.last_p[0],
            p[1] - self.solver.last_p[1],
            p[2] - self.solver.last_p[2],
        ];
        // initial guess, based on how p changed and how the jacobian applies to p
        let mut tmp_nn = [
            -tmp_np[0],
            self.jq[2] * tmp_np[1],
            self.jq[4] * self.c2 as f64 * tmp_np[0] - tmp_np[2],
        ];

        tmp_nn = self.solve_lin_equations(tmp_nn);

        for i in 0..self.solver.z.len() {
            self.solver.z[i] = self.solver.last_z[i] - tmp_nn[i];
        }
        for _plsconverge in 0..500 {
            self.evaluate_nonlinearities(self.solver.z, self.fq);

            self.solver.resmaxabs = 0.;
            for x in &self.solver.residue {
                if x.abs() > self.solver.resmaxabs {
                    self.solver.resmaxabs = x.abs();
                }
            }

            if self.solver.resmaxabs < TOL {
                self.solver.set_extrapolation_origin(p, self.solver.z);
                return;
            }

            // update z with the linsolver according to the residue
            tmp_nn = self.solve_lin_equations(self.solver.residue);
            for i in 0..N_N {
                self.solver.z[i] -= tmp_nn[i];
            }
        }
    }

    #[inline(always)]
    fn evaluate_nonlinearities(&mut self, z: [f64; N_N], fq: [[f64; N_N]; P_LEN]) {
        let mut q = self.solver.p_full;

        q[0] += z[1];
        q[1] += z[0] * fq[1][0];
        q[2] += z[0] * fq[2][0] - z[2];
        q[3] += z[2];
        q[4] += z[0] * 0.5 - z[1];
        q[5] += z[0] * fq[5][0] + z[1] * 3. + z[2];

        let (res1, jq1) = self.solver.eval_opamp(q[0], q[1]);
        let (res2, jq2) = self.solver.eval_opamp(q[2], q[3]);
        let (res3, jq3) = self.solver.eval_diodepair(q[4], q[5], 1e-12, 1.28);

        self.jq[0] = jq1[0];
        self.jq[2] = jq2[0];
        self.jq[4] = jq3[0];

        self.solver.residue = [res1, res2, res3];
    }
    // solve for a value of z, that according to the jacobian should hit the nullspace, so it can be subtracted from the current z
    #[inline(always)]
    fn solve_lin_equations(&mut self, b: [f64; N_N]) -> [f64; N_N] {
        let j00 = -self.fq[1][0];
        let j01 = self.jq[0];

        let j10 = self.jq[2] * self.fq[2][0];
        let j12 = -self.jq[2] - 1.;

        let j20 = self.jq[4] * 0.5 - self.fq[5][0];
        let j21 = -self.jq[4] - 3.;
        let j22 = self.jq[4] - 1.;

        let mut x = [0.; 3];

        x[0] = (b[0] * j12 * j21 + b[1] * j01 * j22 - b[2] * j01 * j12)
            / (j00 * j12 * j21 + j01 * j10 * j22 - j01 * j12 * j20);

        x[1] = (-j00 * x[0] + b[0]) / j01;
        x[2] = (-j20 * x[0] - j21 * x[1] + b[2]) / j22;

        x
    }

    #[inline]
    fn get_output(&self, input: f32, k: f32) -> f32 {
        match self.params.mode.value() {
            SvfMode::LP => self.vout[0], // lowpass
            // the highpass is really quiet for some reason,
            // just multiplied it by 2 here, seems a generally good compromise
            SvfMode::HP => self.vout[2] * 2., // highpass
            SvfMode::BP1 => self.vout[1],     // bandpass
            // the notch isn't limited to the -1 to 1 range like the other modes, not sure how to solve nicely for it currently
            SvfMode::Notch => input + k * self.vout[1], // notch
            //3 => input + 2. * k * self.vout[1], // allpass
            SvfMode::BP2 => k * self.vout[1], // bandpass (normalized peak gain)
                                              // _ => input + 2. * self.vout[1] + k * self.vout[0], // peak / resonator thingy
        }
    }
}