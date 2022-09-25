use crate::{filter::DKSolver, filter_params::FilterParams, utils::AtomicOps};
// use packed_simd::f32x4;
// use core_simd::*;
// use std_float::*;
use std::sync::Arc;

const N_P: usize = 2;
const N_N: usize = 4;
const P_LEN: usize = 8;
const N_OUTS: usize = 1;
const N_STATES: usize = 2;
const TOL: f64 = 1e-5;
pub struct SallenKey {
    filters: [SallenKeyCoreFast; 2],
}

impl SallenKey {
    pub fn new(params: Arc<FilterParams>) -> Self {
        Self {
            filters: [
                SallenKeyCoreFast::new(params.clone()),
                SallenKeyCoreFast::new(params),
            ],
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
pub struct SallenKeyCore {
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

    pub solver: DKSolver<N_N, N_P, P_LEN>,
}
// here we flatten a bunch of stuff to hopefully make it faster
impl SallenKeyCore {
    pub fn new(params: Arc<FilterParams>) -> Self {
        // TODO: pass in proper params
        let fs = params.sample_rate.get();
        let g = (std::f32::consts::PI * 1000. / (fs as f32)).tan();
        let res = 0.1;
        let g_f64 = g as f64;
        let res_f64 = res as f64;

        let pexps = [
            [0., 0.],
            [0., 0.],
            [1., 0.],
            [0., 0.],
            [0., -1.],
            [0., 1.],
            [0., 1.],
            [0., 0.],
        ];
        let fq = [
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [
                -4. * g_f64 * ((res_f64 - 1.) / (4. * res_f64) - 0.25) * (1. / (4. * g_f64) + 0.5),
                0.,
                4. * g_f64 * (1. / (4. * g_f64) + 0.5) - 1.,
                0.,
            ],
            [(res_f64 - 1.) / (4. * res_f64) - 0.25, 0., 0., 0.],
            [-0.25, -1., -(2. * g_f64 + 1.), 0.],
            [1.25, 1., 2. * g_f64 + 1., 1.],
            [0.25, 1., 2. * g_f64 + 1., 0.],
            [0., 0., 0., 1.],
        ];
        let mut a = Self {
            params,
            vout: [0.; 1],
            s: [0.; 2],

            dq: [[0., 1.], [-1., 0.]],
            eq: [0., -2. * g],
            fq,
            pexps,

            a: [[1., 0.], [0., 1.]],
            b: [4. * g, 0.],
            c: [
                [0., 0., -4. * g, 0.],
                [-4. * g * ((res - 1.) / (4. * res) - 0.25), 0., 4. * g, 0.],
            ],

            dy: [[0., 0.]],
            ey: [0.],
            fy: [[(res - 1.) / (4. * res) - 0.25, 0., 0., 0.]],

            solver: DKSolver::new(),
        };
        a.reset();

        a
    }
    pub fn update_matrices(&mut self) {
        let g = self.params.g.get();
        // the model starts to self-oscillate at 0.8
        let res = (self.params.res.value() * 0.79).clamp(0.01, 0.99);
        let g_f64 = g as f64;
        let res_f64 = res as f64;

        self.fq = [
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [
                -4. * g_f64 * ((res_f64 - 1.) / (4. * res_f64) - 0.25) * (1. / (4. * g_f64) + 0.5),
                0.,
                4. * g_f64 * (1. / (4. * g_f64) + 0.5) - 1.,
                0.,
            ],
            [(res_f64 - 1.) / (4. * res_f64) - 0.25, 0., 0., 0.],
            [-0.25, -1., -(2. * g_f64 + 1.), 0.],
            [1.25, 1., 2. * g_f64 + 1., 1.],
            [0.25, 1., 2. * g_f64 + 1., 0.],
            [0., 0., 0., 1.],
        ];

        self.b[0] = 4. * g;

        self.c[0][2] = -4. * g;
        self.c[1][0] = g / res;
        self.c[1][2] = 4. * g;

        self.eq[1] = 2. * g;

        self.fy[0][0] = -0.25 / res;
    }

    pub fn tick_dk(&mut self, input: f32) -> f32 {
        let input = input * (self.params.drive.value());

        let mut p = [0f64; 2];

        p[0] = (self.dq[0][0] * self.s[0] + self.dq[0][1] * self.s[1] + self.eq[0] * input) as f64;
        p[1] = (self.dq[1][0] * self.s[0] + self.dq[1][1] * self.s[1] + self.eq[1] * input) as f64;

        //
        // find nonlinear contributions (solver.z), applying homotopy if it fails to converge
        // self.nonlinear_contribs(p);
        self.homotopy_solver(p);
        self.vout[0] = self.dy[0][0] * self.s[0] + self.dy[0][1] * self.s[1] + self.ey[0] * input;

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
        self.vout[0]
    }

    pub fn homotopy_solver(&mut self, p: [f64; N_P]) {
        self.nonlinear_contribs(p);
        // if the newton solver failed to converge, apply homotopy
        if self.solver.resmaxabs >= TOL {
            // println!("needs homotopy. p: {:?}", p);
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
            // if self.solver.resmaxabs >= TOL {
            //     println!("failed to converge. residue: {:?}", self.solver.residue);

            //     for x in &self.solver.z {
            //         if !x.is_finite() {
            //             panic!("solution contains infinite value");
            //         }
            //     }
            // }
        }
    }

    // uses newton's method to find the nonlinear contributions in the circuit. Not guaranteed to converge
    fn nonlinear_contribs(&mut self, p: [f64; N_P]) -> [f64; N_N] {
        self.solver.set_p(p, &self.pexps);

        let mut tmp_np = [0.; N_P];
        tmp_np[0] = p[0] - self.solver.last_p[0];
        tmp_np[1] = p[1] - self.solver.last_p[1];

        let mut tmp_nn = [0.; N_N];
        for i in 0..N_N {
            for j in 0..N_P {
                tmp_nn[i] += self.solver.jp[i][j] * tmp_np[j];
            }
        }

        tmp_nn = self.solver.solve_linear_equations(tmp_nn);

        for i in 0..self.solver.z.len() {
            self.solver.z[i] = self.solver.last_z[i] - tmp_nn[i];
        }

        for _plsconverge in 0..500 {
            self.evaluate_nonlinearities(self.solver.z, self.fq);

            self.solver.resmaxabs = 0.;
            for x in &self.solver.residue {
                if x.is_finite() {
                    if x.abs() > self.solver.resmaxabs {
                        self.solver.resmaxabs = x.abs();
                    }
                } else {
                    // if any of the residue have become NaN/inf, stop early with big residue
                    self.solver.resmaxabs = 1000.;
                    return self.solver.z;
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
        } else {
            // println!("failed to converge. residue: {:?}", self.solver.residue);
        }
        self.solver.z
    }

    fn evaluate_nonlinearities(&mut self, z: [f64; N_N], fq: [[f64; N_N]; P_LEN]) {
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

        let (res3, jq3) = self.solver.eval_diode(&q[4..6], 1e-15, 1.5);
        let (res4, jq4) = self.solver.eval_diode(&q[6..8], 1e-15, 1.5);

        self.solver.jq[0][0] = jq1[0];
        self.solver.jq[0][1] = jq1[1];
        self.solver.jq[1][2] = jq2[0];
        self.solver.jq[1][3] = jq2[1];

        self.solver.jq[2][4] = jq3[0];
        self.solver.jq[2][5] = jq3[1];
        self.solver.jq[3][6] = jq4[0];
        self.solver.jq[3][7] = jq4[1];

        // update j to the matrix product fq * jq
        for i in 0..self.solver.jq.len() {
            for j in 0..N_N {
                self.solver.j[i][j] = 0.;
                for k in 0..P_LEN {
                    self.solver.j[i][j] += self.solver.jq[i][k] * fq[k][j];
                }
            }
        }
        self.solver.residue = [res1, res2, res3, res4];
    }
    pub fn reset(&mut self) {
        self.s = [0.; 2];
        self.solver.set_p([0.; N_P], &self.pexps);
        self.evaluate_nonlinearities([0.; N_N], self.fq);
        self.solver.set_lin_solver(self.solver.j);
        self.solver.set_jp(&self.pexps);
        self.solver.set_extrapolation_origin([0.; N_P], [0.; N_N]);
    }
}

const N_P2: usize = 2;
const N_N2: usize = 3;
const P_LEN2: usize = 6;
/// this does the same as `SallenKeyCore`, but with most equations simplified to make it faster
pub struct SallenKeyCoreFast {
    pub params: Arc<FilterParams>,
    pub vout: [f32; N_OUTS],
    pub s: [f32; N_STATES],

    // used to find the nonlinear contributions
    eq: [f32; N_P2],
    // pub fq: [[f64; N_N2]; P_LEN2],
    fq20: f64,
    fq22: f64,
    fq30: f64,
    fq40: f64,
    fq42: f64,
    fq50: f64,
    fq52: f64,

    // used to update the capacitor states
    b: [f32; N_STATES],
    c: [[f32; N_N2]; N_STATES],

    // used to find the output values
    fy: [[f32; N_N2]; N_OUTS],

    jq: [f64; 6],

    solver: DKSolver<N_N2, N_P2, P_LEN2>,
}
// here we flatten a bunch of stuff to hopefully make it faster
impl SallenKeyCoreFast {
    pub fn new(params: Arc<FilterParams>) -> Self {
        let fs = params.sample_rate.get();
        let g = (std::f32::consts::PI * 1000. / (fs as f32)).tan();
        let res = 0.1;
        let g_f64 = g as f64;
        let res_f64 = res as f64;

        let mut a = Self {
            params,
            vout: [0.; 1],
            s: [0.; 2],

            eq: [0., 2. * g],
            fq20: (0.25 + 0.5 * g_f64) / res_f64,
            fq22: 2. * g_f64,
            fq30: -0.25 / res_f64,
            fq40: 0.25,
            fq42: (2. * g_f64 + 1.),
            fq50: -1.25,
            fq52: -(2. * g_f64 + 1.),

            b: [4. * g, 0.],
            c: [[0., 0., -4. * g], [g / res, 0., 4. * g]],

            fy: [[-0.25 / res, 0., 0.]],

            jq: [0., -1., 0., -1., 0., 1.],

            solver: DKSolver::new(),
        };
        a.reset();

        a
    }
    pub fn update_matrices(&mut self) {
        let g = self.params.g.get();
        let res = (self.params.res.value() * 0.79).clamp(0.01, 0.99);
        let g_f64 = g as f64;
        let res_f64 = res as f64;

        self.fq30 = -0.25 / res_f64;
        self.fq22 = 2. * g_f64;
        self.fq20 = (0.25 + 0.5 * g_f64) / res_f64;
        self.fq42 = 2. * g_f64 + 1.;
        self.fq52 = -(2. * g_f64 + 1.);

        self.b[0] = 4. * g;

        self.c[0][2] = -4. * g;
        self.c[1][0] = g / res;
        self.c[1][2] = 4. * g;

        self.eq[1] = 2. * g;

        self.fy[0][0] = -0.25 / res;
    }

    pub fn tick_dk(&mut self, input: f32) -> f32 {
        let input = input * (self.params.drive.value());

        // let p = dot(dq, s) + dot(eq, input);
        let mut p = [0f64; 2];
        p[0] = self.s[1] as f64;
        p[1] = (self.s[0] + self.eq[1] * input) as f64;

        // self.nonlinear_contribs(p);
        // find nonlinear contributions (values for solver.z that falls in the null-space described by fq), applying homotopy if it fails to converge
        self.homotopy_solver(p);
        // find output voltage(s)
        self.vout[0] = self.fy[0][0] * self.solver.z[0] as f32;
        // update states
        self.s[0] = self.s[0] + self.b[0] * input + self.solver.z[2] as f32 * self.c[0][2];
        self.s[1] = self.s[1]
            + self.solver.z[0] as f32 * self.c[1][0]
            + self.solver.z[2] as f32 * self.c[1][2];
        self.vout[0]
    }

    pub fn homotopy_solver(&mut self, p: [f64; N_P2]) {
        self.nonlinear_contribs(p);
        // if the newton solver failed to converge, apply homotopy
        if self.solver.resmaxabs >= TOL {
            // println!("needs homotopy. p: {:?}", p);
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
            // this doesn't seem to ever happen anymore
            // if self.solver.resmaxabs >= TOL {
            //     println!("failed to converge. residue: {:?}", self.solver.residue);

            //     for x in &self.solver.z {
            //         if !x.is_finite() {
            //             panic!("solution contains infinite value");
            //         }
            //     }
            // }
        }
    }

    // uses newton's method to find the nonlinear contributions in the circuit. Not guaranteed to converge
    #[inline]
    fn nonlinear_contribs(&mut self, p: [f64; N_P2]) -> [f64; N_N2] {
        self.solver.p_full[2] = p[0];
        self.solver.p_full[4] = -p[1];
        self.solver.p_full[5] = p[1];
        let tmp_np = [p[0] - self.solver.last_p[0], p[1] - self.solver.last_p[1]];

        let mut tmp_nn = [0., self.jq[2] * tmp_np[0], (-self.jq[4] - 1.0) * tmp_np[1]];
        tmp_nn = self.solve_lin_equations(tmp_nn);
        for i in 0..N_N2 {
            self.solver.z[i] = self.solver.last_z[i] - tmp_nn[i];
        }
        for _plsconverge in 0..500 {
            self.evaluate_nonlinearities(self.solver.z);

            self.solver.resmaxabs = 0.;
            for x in &self.solver.residue {
                if x.is_finite() {
                    if x.abs() > self.solver.resmaxabs {
                        self.solver.resmaxabs = x.abs();
                    }
                } else {
                    // if any of the residues have become NaN/inf, stop early with big residue
                    self.solver.resmaxabs = 1000.;
                    return self.solver.z;
                }
            }

            if self.solver.resmaxabs < TOL {
                break;
            }

            // update z with the linsolver according to the residue
            tmp_nn = self.solve_lin_equations(self.solver.residue);
            for i in 0..N_N2 {
                self.solver.z[i] -= tmp_nn[i];
            }
        }
        if self.solver.resmaxabs < TOL {
            // only update this when residue's below tolerance, so homotopy solver can try again with the same initial guess if it failed to converge
            self.solver.set_extrapolation_origin(p, self.solver.z);
        }
        self.solver.z
    }
    // solve for a value of z, that according to the jacobian should hit the nullspace, so it can be subtracted from the current z
    #[inline]
    fn solve_lin_equations(&mut self, b: [f64; N_N2]) -> [f64; N_N2] {
        // let j = self.solver.j;
        let mut x = [0.; 3];
        let j01 = self.jq[0];
        let j10 = self.jq[2] * self.fq20 - self.fq30;
        let j12 = self.jq[2] * self.fq22;
        let j20 = self.jq[4] * self.fq40 - self.fq50;
        let j21 = self.jq[4] + 1.0;
        let j22 = self.jq[4] * self.fq42 - self.fq52;
        // solved the equation `dot_product(self.solver.j, x) = b` for x analytically to get this
        let x0_den = j01 * j10 * j22 - j01 * j12 * j20 + j10 * j21;
        x[0] = (b[0] * j12 * j21 + b[1] * j01 * j22 - b[2] * j01 * j12 + b[1] * j21) / x0_den;

        x[2] = (-j10 * x[0] + b[1]) / j12;
        x[1] = (b[0] + x[2]) / j01;
        x
    }
    #[inline(always)]
    pub fn evaluate_nonlinearities(&mut self, z: [f64; N_N2]) {
        let q = [
            z[1],
            z[2],
            self.solver.p_full[2] + z[0] * self.fq20 + z[2] * self.fq22,
            z[0] * self.fq30,
            self.solver.p_full[4] + z[0] * 0.25 + z[1] + z[2] * self.fq42,
            self.solver.p_full[5] + z[0] * -1.25 - z[1] + z[2] * self.fq52,
        ];

        let (res1, jq1) = self.solver.eval_opamp(q[0], q[1]);
        let (res2, jq2) = self.solver.eval_opamp(q[2], q[3]);
        let (res3, jq3) = self.solver.eval_diodepair(q[4], q[5], 1e-15, 1.7);

        self.solver.residue = [res1, res2, res3];
        self.jq[0] = jq1[0];
        self.jq[2] = jq2[0];
        self.jq[4] = jq3[0];
    }

    pub fn reset(&mut self) {
        self.s = [0.; 2];
        self.solver.p_full = [0.; P_LEN2];
        self.evaluate_nonlinearities([0.; N_N2]);
        self.solver.set_extrapolation_origin([0.; N_P2], [0.; N_N2]);
    }
}