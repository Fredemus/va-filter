use crate::{
    filter::DKSolver,
    filter_params_nih::{FilterParams, SvfMode},
    utils::AtomicOps,
};
// use packed_simd::f32x4;
use core_simd::*;
use std::sync::Arc;
use std_float::*;

// FIXME: something's wrong, reso doesn't do what it should huge DC offset

use core_simd::f32x4;

const N_P: usize = 3;
const N_N: usize = 4;
const P_LEN: usize = 8;
const N_OUTS: usize = 2;
const N_STATES: usize = 2;
const TOL: f32 = 1e-4;
/// 2-pole state-variable filter 
pub struct Svf {
    pub params: Arc<FilterParams>,
    pub vout: [f32; N_OUTS],
    pub s: [f32; N_STATES],

    // used to find the nonlinear contributions
    dq: [[f32; 2]; N_P],
    eq: [f32; N_P],
    pub fq: [[f32; 4]; P_LEN],
    // dq, eq are actually much larger, pexps are used to reduce them
    pexps: [[f32; N_P]; P_LEN],

    // used to update the capacitor states
    a: [[f32; 2]; N_STATES],
    b: [f32; N_STATES],
    c: [[f32; 4]; N_STATES],

    // used to find the output values
    // dy: [[f32; 2]; 2],
    dy: [[f32; 2]; N_OUTS],
    ey: [f32; N_OUTS],
    fy: [[f32; 4]; N_OUTS],

    solver: DKSolver<N_N, N_P, P_LEN>,
}

impl Svf {
    pub fn new(params: Arc<FilterParams>) -> Self {
        // TODO: pass in proper params
        let fs = params.sample_rate.get();
        let g = (std::f32::consts::PI * 1000. / (fs as f32)).tan();
        let res = 0.1;

        let pexps = [
            [1., 0., 0.],
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 0., 0.],
            [0., 0., 0.],
        ];
        let fq = [
            [g, -1., 0., 0.],
            [0., 1., 0., 0.],
            [0., g, -1., 0.],
            [0., 0., 1., 0.],
            [0., -1., 0., 0.],
            [1., res, 2., 1.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
        ];
        let mut a = Self {
            params,
            vout: [0.; N_OUTS],
            s: [0.; 2],

            dq: [[-1., 0.], [0., -1.], [0., 0.]],
            eq: [0., 0., 1.],
            fq,
            pexps,

            a: [[1., 0.], [0., 1.]],
            b: [0., 0.],
            c: [[-2. * g, 0., 0., 0.], [0., -2. * g, 0., 0.]],

            dy: [[0., 0.], [0., 0.]],
            ey: [0., 0.],
            fy: [[0., 0., 1., 0.], [0., 1., 0., 0.]],

            solver: DKSolver::new(),
        };
        a.solver.set_p([0.; N_P], &pexps);
        a.evaluate_nonlinearities([0.; N_N], fq);
        a.solver.set_lin_solver(a.solver.j);
        a.solver.set_jp(&pexps);
        a.solver
            .set_extrapolation_origin([0.; N_P], [0.; N_N], a.solver.jp);

        a
    }
    pub fn update_matrices(&mut self) {
        // let fs = self.params.sample_rate.get();
        let g = self.params.g.get();
        let res = self.params.zeta.get();

        self.fq[0][0] = g;
        self.fq[2][1] = g;
        self.fq[5][1] = res;

        self.c[0][0] = -2. * g;
        self.c[1][1] = -2. * g;
    }
    // TODO: pls roll the updates into for loops
    pub fn tick_dk(&mut self, input: f32x4) -> f32x4 {
        let input_l = input.as_array()[0];
        let input = input_l * (self.params.drive.value);

        // let p = dot(dq, s) + dot(eq, input);
        let mut p = [0.; N_P];
        // find the
        p[0] = self.dq[0][0] * self.s[0] + self.dq[0][1] * self.s[1] + self.eq[0] * input;
        p[1] = self.dq[1][0] * self.s[0] + self.dq[1][1] * self.s[1] + self.eq[1] * input;
        p[2] = self.dq[2][0] * self.s[0] + self.dq[2][1] * self.s[1] + self.eq[2] * input;

        //
        // self.nonlinear_contribs(p);
        // find nonlinear contributions (solver.z), applying homotopy if it fails to converge
        self.homotopy_solver(p);
        // self.vout = dot(dy, s) + dot(ey, input) + dot(fy, self.solver.z)
        // TODO: add in fy * z
        self.vout[0] = self.dy[0][0] * self.s[0] + self.dy[0][1] * self.s[1] + self.ey[0] * input;
        self.vout[1] = self.dy[1][0] * self.s[0] + self.dy[1][1] * self.s[1] + self.ey[1] * input;

        // self.vout[1] = self.dy[1][0] * self.s[0] + self.dy[1][1] * self.s[1] + self.ey[1] * input;
        for i in 0..N_OUTS {
            for j in 0..4 {
                self.vout[i] += self.solver.z[j] * self.fy[i][j];
            }
        }
        let s1_update = self.a[1][0] * self.s[0] + self.a[1][1] * self.s[1];
        self.s[0] = self.a[0][0] * self.s[0] + self.a[0][1] * self.s[1] + self.b[0] * input;
        self.s[1] = s1_update + self.b[1] * input;
        // correct formula: self.s[1] = self.a[1][0] * self.s[0] + self.a[1][1] * self.s[1] +  self.b[1] * input + self.solver.z[1] * self.c[1][1]
        for i in 0..2 {
            for j in 0..4 {
                self.s[i] += self.solver.z[j] * self.c[i][j];
            }
        }
        let out = self.get_output(input, self.params.zeta.get());
        // let out = self.vout[0];
        f32x4::from_array([out, out, 0., 0.])
        // self.s = dot(a, s) + dot(b, input) + dot(c, self.solver.z);
    }

    fn homotopy_solver(&mut self, p: [f32; N_P]) {
        self.nonlinear_contribs(p);
        // if the newton solver failed to converge, apply homotopy
        if !(self.solver.resmaxabs < TOL) {
            println!("needs homotopy");
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
        }
    }

    // uses newton's method to find the nonlinear contributions in the circuit. Not guaranteed to converge
    fn nonlinear_contribs(&mut self, p: [f32; N_P]) {
        self.solver.set_p(p, &self.pexps);

        // self.solver.tmp_np[0] =  self.solver.tmp_np[0] - self.solver.last_p[0];
        // self.solver.tmp_np[1] = self.solver.tmp_np[1] - self.solver.last_p[1];
        self.solver.tmp_np[0] = p[0] - self.solver.last_p[0];
        self.solver.tmp_np[1] = p[1] - self.solver.last_p[1];
        self.solver.tmp_np[2] = p[2] - self.solver.last_p[2];
        // dbg!(self.solver.tmp_nn);
        for i in 0..N_N {
            self.solver.tmp_nn[i] = 0.;
            for j in 0..self.solver.tmp_np.len() {
                self.solver.tmp_nn[i] += self.solver.last_jp[i][j] * self.solver.tmp_np[j];
            }
        }

        self.solver.tmp_nn = self.solver.solve_linear_equation(self.solver.tmp_nn);

        // self.solver.z = self.solver.last_z - self.solver.tmp_nn;
        for i in 0..self.solver.z.len() {
            self.solver.z[i] = self.solver.last_z[i] - self.solver.tmp_nn[i];
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
                    // if any of the residue have become NaN/inf, stop early.
                    // If using the homotopy solver, it will kick in and find an alternate, slower path to convergence
                    // self.solver.resmaxabs = ;
                    // return self.solver.z;
                    return;
                }
            }

            self.solver.set_lin_solver(self.solver.j);
            if self.solver.resmaxabs < TOL {
                break;
            }

            // update z with the linsolver according to the residue
            self.solver.tmp_nn = self.solver.solve_linear_equation(self.solver.residue);

            for i in 0..self.solver.z.len() {
                self.solver.z[i] = self.solver.z[i] - self.solver.tmp_nn[i];
            }
            // self.solver.z = self.solver.z - self.solver.tmp_nn;
        }
        // self.solver.resmaxabs = resmaxabs;
        if self.solver.resmaxabs < TOL {
            self.solver.set_jp(&self.pexps);
            self.solver
                .set_extrapolation_origin(p, self.solver.z, self.solver.jp);
        }
        // else {
        // panic!("failed to converge. residue: {:?}", self.solver.residue);
        // println!("failed to converge. residue: {:?}", self.solver.residue);
        // }
        // return self.solver.z;
    }

    fn evaluate_nonlinearities(&mut self, z: [f32; N_N], fq: [[f32; N_N]; P_LEN]) {
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

        let (res3, jq3) = self.solver.eval_diode(&q[4..6]);
        let (res4, jq4) = self.solver.eval_diode(&q[6..8]);

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
    // highpass and notch doesn't work right, likely because `input` isn't quite defined right. Prolly doesn't need to be subtracted?
    fn _get_output_old(&self, input: f32, k: f32) -> f32 {
        match self.params.mode.value() {
            SvfMode::LP => self.vout[0],                            // lowpass
            SvfMode::HP => input - k * self.vout[1] - self.vout[0], // highpass
            SvfMode::BP1 => self.vout[1],                           // bandpass
            SvfMode::Notch => input - k * self.vout[1],             // notch
            //3 => input - 2. * k * self.vout[1], // allpass
            SvfMode::BP2 => k * self.vout[1], // bandpass (normalized peak gain)
                                              // _ => input - f32x4::splat(2.) * self.vout[1] - k * self.vout[0], // peak / resonator thingy
        }
    }
    // TODO: should prolly redo model to regain the nicer simpler equations here
    fn get_output(&self, input: f32, k: f32) -> f32 {
        match self.params.mode.value() {
            SvfMode::LP => self.vout[0], // lowpass
            SvfMode::HP => -0.5 * input - 0.5 * k * self.vout[1] - self.vout[0], // highpass
            SvfMode::BP1 => self.vout[1], // bandpass
            SvfMode::Notch => -0.5 * input - 0.5 * k * self.vout[1], // notch
            //3 => input - 2. * k * self.vout[1], // allpass
            SvfMode::BP2 => k * self.vout[1], // bandpass (normalized peak gain)
                                              // _ => input - f32x4::splat(2.) * self.vout[1] - k * self.vout[0], // peak / resonator thingy
        }
    }
}
#[test]
fn test_stepresponse() {
    let should_update_filter = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let params = Arc::new(FilterParams::new(should_update_filter.clone()));
    // params.cutoff.set_plain_value(1000.);
    params.sample_rate.set(44100.);
    params.update_g(1000.);
    params.zeta.set(0.1);
    let mut filt = Svf::new(params.clone());
    filt.update_matrices();
    // println!("should be 1.5889e-04: {}", filt.fq[1][1]);
    let mut out = [0.; 10];
    for i in 0..10 {
        println!("sample {i}");
        filt.tick_dk(f32x4::splat(1.0));
        out[i] = filt.vout[0];
        // println!("val lp: {}", filt.vout[0]);
        // println!("filter state: {:?}", filt.s);
        // if filt.vout[0] < 0. {
        //     panic!("sample {} got negative", i)
        // }
    }
    dbg!(out);
}
