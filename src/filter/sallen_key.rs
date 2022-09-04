use crate::{filter::DKSolver, filter_params_nih::FilterParams, utils::AtomicOps};
// use packed_simd::f32x4;
use core_simd::*;
use std::sync::Arc;
use std_float::*;

use core_simd::f32x4;

const N_P: usize = 2;
const N_N: usize = 4;
const P_LEN: usize = 8;
const N_OUTS: usize = 1;
const N_STATES: usize = 2;
const TOL: f64 = 1e-5;

pub struct SallenKey {
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
// here we flatten a bunch of stuff to hopefully make it faster
impl SallenKey {
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
        let res = (self.params.res.value * 0.8).clamp(0.01, 0.99);
        let g_f64 = g as f64;
        let res_f64 = res as f64;
        // println!("res: {res}");
        // println!("res: {g}");
        // TODO: no need to set the entire matrix but lazy rn
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
            [-0.25, -1., -2. * g_f64 - 1., 0.],
            [1.25, 1., 2. * g_f64 + 1., 1.],
            [0.25, 1., 2. * g_f64 + 1., 0.],
            [0., 0., 0., 1.],
        ];

        self.b[0] = 4. * g;

        self.c[0][2] = -4. * g;
        self.c[1][0] = -4. * g * ((res - 1.) / (4. * res) - 0.25);
        self.c[1][2] = 4. * g;

        self.eq[1] = -2. * g;

        self.fy[0][0] = (res - 1.) / (4. * res) - 0.25;
    }

    pub fn tick_dk(&mut self, input: f32x4) -> f32x4 {
        let input_l = input.as_array()[0];
        let input = input_l * (self.params.drive.value);

        // let p = dot(dq, s) + dot(eq, input);
        let mut p = [0f64; 2];
        // find the
        p[0] = (self.dq[0][0] * self.s[0] + self.dq[0][1] * self.s[1] + self.eq[0] * input) as f64;
        p[1] = (self.dq[1][0] * self.s[0] + self.dq[1][1] * self.s[1] + self.eq[1] * input) as f64;
        // p[2] = self.dq[2][0] * self.s[0] + self.dq[2][1] * self.s[1] + self.eq[2] * input;

        //
        // self.nonlinear_contribs(p);
        // find nonlinear contributions (solver.z), applying homotopy if it fails to converge
        self.homotopy_solver(p);
        // self.vout = dot(dy, s) + dot(ey, input) + dot(fy, self.solver.z)
        // TODO: add in fy * z
        self.vout[0] = self.dy[0][0] * self.s[0] + self.dy[0][1] * self.s[1] + self.ey[0] * input;

        // self.vout[1] = self.dy[1][0] * self.s[0] + self.dy[1][1] * self.s[1] + self.ey[1] * input;
        for i in 0..N_OUTS {
            for j in 0..N_N {
                self.vout[i] += self.solver.z[j] as f32 * self.fy[i][j];
            }
        }
        let s1_update = self.a[1][0] * self.s[0] + self.a[1][1] * self.s[1];
        self.s[0] = self.a[0][0] * self.s[0] + self.a[0][1] * self.s[1] + self.b[0] * input;
        self.s[1] = s1_update + self.b[1] * input;
        // correct formula: self.s[1] = self.a[1][0] * self.s[0] + self.a[1][1] * self.s[1] +  self.b[1] * input + self.solver.z[1] * self.c[1][1]
        for i in 0..N_STATES {
            for j in 0..N_N {
                self.s[i] += self.solver.z[j] as f32 * self.c[i][j];
            }
        }
        // let out = self.get_output(input, self.params.zeta.get());
        let out = self.vout[0];
        f32x4::from_array([out, out, 0., 0.])
        // self.s = dot(a, s) + dot(b, input) + dot(c, self.solver.z);
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
            if self.solver.resmaxabs >= TOL {
                println!("failed to converge. residue: {:?}", self.solver.residue);

                for x in &self.solver.z {
                    if !x.is_finite() {
                        panic!("solution contains infinite value");
                    }
                }
            }
        }
    }

    // uses newton's method to find the nonlinear contributions in the circuit. Not guaranteed to converge
    fn nonlinear_contribs(&mut self, p: [f64; N_P]) -> [f64; N_N] {
        self.solver.set_p(p, &self.pexps);

        // self.solver.tmp_np[0] =  self.solver.tmp_np[0] - self.solver.last_p[0];
        // self.solver.tmp_np[1] = self.solver.tmp_np[1] - self.solver.last_p[1];
        self.solver.tmp_np[0] = p[0] - self.solver.last_p[0];
        self.solver.tmp_np[1] = p[1] - self.solver.last_p[1];
        // dbg!(self.solver.tmp_nn);
        for i in 0..N_N {
            self.solver.tmp_nn[i] = 0.;
            for j in 0..N_P {
                self.solver.tmp_nn[i] += self.solver.last_jp[i][j] * self.solver.tmp_np[j];
            }
        }

        self.solver.tmp_nn = self.solver.solve_linear_equations(self.solver.tmp_nn);

        // self.solver.z = self.solver.last_z - self.solver.tmp_nn;
        for i in 0..self.solver.z.len() {
            self.solver.z[i] = self.solver.last_z[i] - self.solver.tmp_nn[i];
        }
        // let mut resmaxabs = 0.;
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
                    // self.solver.resmaxabs = ;
                    return self.solver.z;
                }
            }

            self.solver.set_lin_solver(self.solver.j);
            if self.solver.resmaxabs < TOL {
                break;
            }

            // update z with the linsolver according to the residue
            self.solver.tmp_nn = self.solver.solve_linear_equations(self.solver.residue);

            for i in 0..self.solver.z.len() {
                self.solver.z[i] = self.solver.z[i] - self.solver.tmp_nn[i];
            }
            // self.solver.z = self.solver.z - self.solver.tmp_nn;
        }
        if self.solver.resmaxabs < TOL {
            self.solver.set_jp(&self.pexps);
            self.solver
                .set_extrapolation_origin(p, self.solver.z, self.solver.jp);
        } else {
            // panic!("failed to converge. residue: {:?}", self.solver.residue);
            // println!("failed to converge. residue: {:?}", self.solver.residue);
        }
        self.solver.z
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
        // TODO: could this be done differently?
        let (res1, jq1) = self.solver.eval_opamp(&q[0..2]);
        let (res2, jq2) = self.solver.eval_opamp(&q[2..4]);

        let (res3, jq3) = self.solver.eval_diode(&q[4..6], 1e-15, 1.5);
        let (res4, jq4) = self.solver.eval_diode(&q[6..8], 1e-15, 1.5);

        // TODO: consider simplifying jq
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
        self.solver
            .set_extrapolation_origin([0.; N_P], [0.; N_N], self.solver.jp);
    }
}

const N_P2: usize = 2;
const N_N2: usize = 3;
const P_LEN2: usize = 6;
pub struct SallenKeyFast {
    pub params: Arc<FilterParams>,
    pub vout: [f32; N_OUTS],
    pub s: [f32; N_STATES],

    // used to find the nonlinear contributions
    dq: [[f32; N_STATES]; N_P2],
    eq: [f32; N_P2],
    pub fq: [[f64; N_N2]; P_LEN2],
    // dq, eq are actually much larger, pexps are used to reduce them
    pexps: [[f64; N_P2]; P_LEN2],

    // used to update the capacitor states
    a: [[f32; N_STATES]; N_STATES],
    b: [f32; N_STATES],
    c: [[f32; N_N2]; N_STATES],

    // used to find the output values
    // dy: [[f32; 2]; 2],
    dy: [[f32; N_STATES]; N_OUTS],
    ey: [f32; N_OUTS],
    fy: [[f32; N_N2]; N_OUTS],

    solver: DKSolver<N_N2, N_P2, P_LEN2>,
}
// here we flatten a bunch of stuff to hopefully make it faster
impl SallenKeyFast {
    pub fn new(params: Arc<FilterParams>) -> Self {
        // TODO: pass in proper params
        let fs = params.sample_rate.get();
        let g = (std::f32::consts::PI * 1000. / (fs as f32)).tan();
        let res = 0.1;
        let g_f64 = g as f64;
        let res_f64 = res as f64;

        let pexps = [[0., 0.], [0., 0.], [1., 0.], [0., 0.], [0., -1.], [0., 1.]];
        let fq = [
            [0., 1., 0.],
            [0., 0., 1.],
            [(0.25 + 0.5 * g_f64) / res_f64, 0., 2. * g_f64],
            [-0.25 / res_f64, 0., 0.],
            [0.25, 1., (2. * g_f64 + 1.)],
            [-1.25, -1., -(2. * g_f64 + 1.)],
        ];
        let mut a = Self {
            params,
            vout: [0.; 1],
            s: [0.; 2],

            dq: [[0., 1.], [1., 0.]],
            eq: [0., 2. * g],
            fq,
            pexps,

            a: [[1., 0.], [0., 1.]],
            b: [4. * g, 0.],
            c: [
                [0., 0., -4. * g],
                [-4. * g * ((res - 1.) / (4. * res) - 0.25), 0., 4. * g],
            ],

            dy: [[0., 0.]],
            ey: [0.],
            fy: [[(res - 1.) / (4. * res) - 0.25, 0., 0.]],

            solver: DKSolver::new(),
        };
        a.reset();

        a
    }
    pub fn update_matrices(&mut self) {
        let g = self.params.g.get();
        let res = (self.params.res.value * 0.8).clamp(0.01, 0.99);
        let g_f64 = g as f64;
        let res_f64 = res as f64;

        self.fq[3][0] = -0.25 / res_f64;
        self.fq[2][2] = 2. * g_f64;
        self.fq[2][0] = (0.25 + 0.5 * g_f64) / res_f64;
        self.fq[4][2] = 2. * g_f64 + 1.;
        self.fq[5][2] = -(2. * g_f64 + 1.);

        self.b[0] = 4. * g;

        self.c[0][2] = -4. * g;
        self.c[1][0] = g / res;
        self.c[1][2] = 4. * g;

        self.eq[1] = 2. * g;

        self.fy[0][0] = (res - 1.) / (4. * res) - 0.25;
    }

    pub fn tick_dk(&mut self, input: f32x4) -> f32x4 {
        let input_l = input.as_array()[0];
        let input = input_l * (self.params.drive.value);

        // let p = dot(dq, s) + dot(eq, input);
        let mut p = [0f64; 2];
        // find the
        p[0] = (self.dq[0][0] * self.s[0] + self.dq[0][1] * self.s[1] + self.eq[0] * input) as f64;
        p[1] = (self.dq[1][0] * self.s[0] + self.dq[1][1] * self.s[1] + self.eq[1] * input) as f64;
        // p[2] = self.dq[2][0] * self.s[0] + self.dq[2][1] * self.s[1] + self.eq[2] * input;

        //
        // self.nonlinear_contribs(p);
        // find nonlinear contributions (solver.z), applying homotopy if it fails to converge
        self.homotopy_solver(p);
        self.vout[0] = self.dy[0][0] * self.s[0] + self.dy[0][1] * self.s[1] + self.ey[0] * input;

        // self.vout[1] = self.dy[1][0] * self.s[0] + self.dy[1][1] * self.s[1] + self.ey[1] * input;
        for i in 0..N_OUTS {
            for j in 0..N_N2 {
                self.vout[i] += self.solver.z[j] as f32 * self.fy[i][j];
            }
        }
        let s1_update = self.a[1][0] * self.s[0] + self.a[1][1] * self.s[1];
        self.s[0] = self.a[0][0] * self.s[0] + self.a[0][1] * self.s[1] + self.b[0] * input;
        self.s[1] = s1_update + self.b[1] * input;
        // correct formula: self.s[1] = self.a[1][0] * self.s[0] + self.a[1][1] * self.s[1] +  self.b[1] * input + self.solver.z[1] * self.c[1][1]
        for i in 0..N_STATES {
            for j in 0..N_N2 {
                self.s[i] += self.solver.z[j] as f32 * self.c[i][j];
            }
        }
        // let out = self.get_output(input, self.params.zeta.get());
        let out = self.vout[0];
        f32x4::from_array([out, out, 0., 0.])
        // self.s = dot(a, s) + dot(b, input) + dot(c, self.solver.z);
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
                        // no values between a and best_a. This could probably be simplified
                        break;
                    }
                    a = new_a;
                }
            }
            if self.solver.resmaxabs >= TOL {
                println!("failed to converge. residue: {:?}", self.solver.residue);

                for x in &self.solver.z {
                    if !x.is_finite() {
                        panic!("solution contains infinite value");
                    }
                }
            }
        }
    }

    // uses newton's method to find the nonlinear contributions in the circuit. Not guaranteed to converge
    fn nonlinear_contribs(&mut self, p: [f64; N_P2]) -> [f64; N_N2] {
        self.solver.p_full[2] = p[0];
        self.solver.p_full[4] = -p[1];
        self.solver.p_full[5] = p[1];

        self.solver.tmp_np[0] = p[0] - self.solver.last_p[0];
        self.solver.tmp_np[1] = p[1] - self.solver.last_p[1];
        for i in 0..N_N2 {
            self.solver.tmp_nn[i] = 0.;
            for j in 0..N_P2 {
                self.solver.tmp_nn[i] += self.solver.last_jp[i][j] * self.solver.tmp_np[j];
            }
        }
        self.solve_lin_equations_new(self.solver.tmp_nn);
        for i in 0..self.solver.z.len() {
            self.solver.z[i] = self.solver.last_z[i] - self.solver.tmp_nn[i];
        }
        // let mut resmaxabs = 0.;
        for _plsconverge in 0..500 {
            self.evaluate_nonlinearities(self.solver.z);

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

            // self.solver.set_lin_solver(self.solver.j);
            if self.solver.resmaxabs < TOL {
                break;
            }

            // update z with the linsolver according to the residue
            // let test = self.solver.solve_linear_equations(self.solver.residue);
            self.solve_lin_equations_new(self.solver.residue);
            for i in 0..self.solver.z.len() {
                self.solver.z[i] = self.solver.z[i] - self.solver.tmp_nn[i];
            }
        }
        if self.solver.resmaxabs < TOL {
            self.solver.set_jp(&self.pexps);
            self.solver
                .set_extrapolation_origin(p, self.solver.z, self.solver.jp);
        } else {
            // panic!("failed to converge. residue: {:?}", self.solver.residue);
            // println!("failed to converge. residue: {:?}", self.solver.residue);
        }
        self.solver.z
    }

    fn solve_lin_equations_new(&mut self, b: [f64; N_N2]) -> bool {
        let j = self.solver.j;
        let mut x = [0.; 3];
        // if j[1][2] == 0.0 || j[0][1] == 0.0 || x0_den == 0.0 {
        //     // dbg!("bad denom");
        //     // Jacobian for a nonlin opamp contains a 0 value, making it impossible to iterate with
        //     // something else must be done, reuse old solution or maybe end nonlin_contribs and fallback to homotopy
        //     // ^ no longer relevant after changes to jacobians
        //     return false;
        // }
        // solved the equation dot_product(self.solver.j, x) = b for x analytically to get this
        let x0_den = j[0][1] * j[1][0] * j[2][2] - j[0][1] * j[1][2] * j[2][0] + j[1][0] * j[2][1];
        x[0] = (b[0] * j[1][2] * j[2][1] + b[1] * j[0][1] * j[2][2] - b[2] * j[0][1] * j[1][2]
            + b[1] * j[2][1])
            / x0_den;

        x[2] = (-j[1][0] * x[0] + b[1]) / j[1][2];
        x[1] = (b[0] + x[2]) / j[0][1];
        self.solver.tmp_nn = x;
        true
    }

    fn evaluate_nonlinearities(&mut self, z: [f64; N_N2]) {
        let fq = self.fq;
        //  q = p_full + dot_product(z, fq)
        let mut q = self.solver.p_full;

        q[0] += z[1];
        q[1] += z[2];
        q[2] += z[0] * fq[2][0] + z[2] * fq[2][2];
        q[3] += z[0] * fq[3][0];
        q[4] += z[0] * 0.25 + z[1] + z[2] * fq[4][2];
        q[5] += z[0] * -1.25 - z[1] + z[2] * fq[5][2];
        // TODO: re-add tuple returning evals, doesn't seem to change performance
        // let (res1, jq1) = self.solver.eval_opamp(&q[0..2]);
        // let (res2, jq2) = self.solver.eval_opamp(&q[2..4]);
        // let (res3, jq3) = self.solver.eval_diode(&q[4..6], 1e-15, 1.5);
        // let (res4, jq4) = self.solver.eval_diode(&q[6..8], 1e-15, 1.5);

        let res_j_1 = self.solver.eval_opamp_arr(&q[0..2]);
        let res_j_2 = self.solver.eval_opamp_arr(&q[2..4]);
        let res_j_3 = self.solver.eval_diodepair_arr(&q[4..6], 1e-15, 1.5);

        self.solver.residue = [res_j_1[0], res_j_2[0], res_j_3[0]];
        // self.solver.residue = [res1, res2, res3, res4];

        self.solver.jq[0][0] = res_j_1[1];
        self.solver.jq[1][2] = res_j_2[1];
        self.solver.jq[2][4] = res_j_3[1];
        // self.solver.jq[0][1] = -1.0;
        // self.solver.jq[1][3] = -1.0;
        // self.solver.jq[2][5] = -1.0;
        // this sets solver.j to the matrix product of solver.jq and fq. A lot of values in those are 0, so those calculations have been removed
        // self.solver.j[0][0] = self.solver.jq[0][0] * fq[0][0] + self.solver.jq[0][1] * fq[1][0]; // <- always 0
        self.solver.j[0][1] = self.solver.jq[0][0]; // fq[0][1] is always 0
        self.solver.j[0][2] = -1.0; // self.solver.jq[0][1] * fq[1][2] is always -1, move somewhere else
                                    // self.solver.j[0][3] = self.solver.jq[0][0] * fq[0][3] + self.solver.jq[0][1] * fq[1][3];  // <- always 0

        self.solver.j[1][0] = self.solver.jq[1][2] * fq[2][0] - fq[3][0];
        // self.solver.j[1][1] = self.solver.jq[1][2] * fq[2][1] + self.solver.jq[1][3] * fq[3][1]; // <- always 0
        self.solver.j[1][2] = self.solver.jq[1][2] * fq[2][2];
        // self.solver.j[1][3] = self.solver.jq[1][2] * fq[2][3] + self.solver.jq[1][3] * fq[3][3];  // <- always 0

        self.solver.j[2][0] = self.solver.jq[2][4] * fq[4][0] - fq[5][0];
        self.solver.j[2][1] = self.solver.jq[2][4] + 1.0;
        self.solver.j[2][2] = self.solver.jq[2][4] * fq[4][2] - fq[5][2];
    }
    pub fn reset(&mut self) {
        self.s = [0.; 2];
        self.solver.set_p([0.; N_P2], &self.pexps);
        self.evaluate_nonlinearities([0.; N_N2]);
        self.solver.set_lin_solver(self.solver.j);
        self.solver.set_jp(&self.pexps);
        self.solver
            .set_extrapolation_origin([0.; N_P2], [0.; N_N2], self.solver.jp);
    }
}

#[test]
fn test_stepresponse() {
    let should_update_filter = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let params = Arc::new(FilterParams::new(should_update_filter.clone()));
    // params.cutoff.set_plain_value(1000.);
    params.sample_rate.set(44100.);
    params.update_g(10000.);
    params.zeta.set(0.1);
    let mut filt = SallenKeyFast::new(params.clone());
    filt.update_matrices();
    // println!("should be 1.5889e-04: {}", filt.fq[1][1]);
    let mut out = [0.; 10];
    for i in 0..10 {
        println!("sample {i}");
        filt.tick_dk(f32x4::splat(-25.0));
        out[i] = filt.vout[0];
    }
    dbg!(out);
}
