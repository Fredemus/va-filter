// use crate::filter_parameters::FilterParameters;
// use crate::filter_params_nih::{FilterParams, SvfMode};
use crate::{
    filter_params_nih::{FilterParams, SvfMode},
    utils::AtomicOps,
};
// use packed_simd::f32x4;
use core_simd::*;
use std::sync::Arc;
use std_float::*;

/// cheap tanh to make the filter faster.
// from a quick look it looks extremely good, max error of ~0.0002 or .02%
// the error of 1 - tanh_levien^2 as the derivative is about .06%
#[inline]
pub fn tanh_levien(x: f32x4) -> f32x4 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let a = x + (f32x4::splat(0.16489087) * x3) + (f32x4::splat(0.00985468) * x5);
    // println!("a: {:?}, b: {:?}", a, b);
    a / (f32x4::splat(1.0) + (a * a)).sqrt()
}
// TODO: scalar exp and ln should really be replaced when portable_simd adds them
#[inline]
fn simd_cosh(x: f32x4) -> f32x4 {
    (f32x4::from_array(x.to_array().map(f32::exp))
        + f32x4::from_array((-x).to_array().map(f32::exp)))
        / f32x4::splat(2.)
}
#[inline]
fn simd_asinh(x: f32x4) -> f32x4 {
    let mask = x.simd_gt(f32x4::splat(0.));
    // let val = (x.abs() + ((x * x) + f32x4::splat(1.0)).sqrt()).ln();
    let mut val = x.abs() + ((x * x) + f32x4::splat(1.0)).sqrt();
    val = f32x4::from_array(val.to_array().map(f32::ln));
    // this be equivalent to copysign, preserving the sign of the input. Untested
    mask.select(val, -val)
}

#[allow(dead_code)]
#[derive(PartialEq, Clone, Copy)]
enum EstimateSource {
    State,               // use current state
    PreviousVout,        // use z-1 of Vout
    LinearStateEstimate, // use linear estimate of future state
    LinearVoutEstimate,  // use linear estimate of Vout
}
pub struct LadderFilter {
    pub params: Arc<FilterParams>,

    vout: [f32x4; 4],
    s: [f32x4; 4],
}
#[allow(dead_code)]
impl LadderFilter {
    pub fn new(params: Arc<FilterParams>) -> Self {
        Self {
            params,
            vout: [f32x4::splat(0.); 4],
            s: [f32x4::splat(0.); 4],
        }
    }
    fn get_estimate(&mut self, n: usize, estimate: EstimateSource, input: f32x4) -> f32x4 {
        // if we ask for an estimate based on the linear filter, we have to run it
        if estimate == EstimateSource::LinearStateEstimate
            || estimate == EstimateSource::LinearVoutEstimate
        {
            self.run_filter_linear(input);
        }
        match estimate {
            EstimateSource::State => self.s[n],
            EstimateSource::PreviousVout => self.vout[n],
            EstimateSource::LinearStateEstimate => f32x4::splat(2.) * self.vout[n] - self.s[n],
            EstimateSource::LinearVoutEstimate => self.vout[n],
        }
    }
    #[inline(always)]
    fn update_state(&mut self) {
        let two = f32x4::splat(2.);
        self.s[0] = two * self.vout[0] - self.s[0];
        self.s[1] = two * self.vout[1] - self.s[1];
        self.s[2] = two * self.vout[2] - self.s[2];
        self.s[3] = two * self.vout[3] - self.s[3];
    }
    // nonlinear ladder filter function with distortion, solved with Mystran's fixed-pivot method.
    fn run_filter_pivotal(&mut self, input: f32x4) -> f32x4 {
        let mut a: [f32x4; 5] = [f32x4::splat(1.); 5];
        // let base = [input, self.s[0], self.s[1], self.s[2], self.s[3]];
        let g = f32x4::splat(self.params.g.get());
        let k = f32x4::splat(self.params.k_ladder.get());
        let base = [
            input - k * self.s[3],
            self.s[0],
            self.s[1],
            self.s[2],
            self.s[3],
        ];
        // a[n] is the fixed-pivot approximation for tanh()
        for n in 0..base.len() {
            // hopefully this should cook down to the original when not 0,
            // and 1 when 0
            let mask = base[n].simd_ne(f32x4::splat(0.));
            a[n] = tanh_levien(base[n]) / base[n];
            // since the line above can become NaN or other stuff when a value in base[n] is 0,
            // replace values where a[n] is 0.
            a[n] = mask.select(a[n], f32x4::splat(1.));
        }
        // denominators of solutions of individual stages. Simplifies the math a bit
        let one = f32x4::splat(1.);
        let g0 = one / (one + g * a[1]);
        let g1 = one / (one + g * a[2]);
        let g2 = one / (one + g * a[3]);
        let g3 = one / (one + g * a[4]);
        // these are factored out of the feedback solution. Makes the math easier to read
        let f3 = g * a[3] * g3;
        let f2 = g * a[2] * g2 * f3;
        let f1 = g * a[1] * g1 * f2;
        let f0 = g * g0 * f1;
        // outputs a 24db filter
        self.vout[3] = (f0 * input * a[0]
            + f1 * g0 * self.s[0]
            + f2 * g1 * self.s[1]
            + f3 * g2 * self.s[2]
            + g3 * self.s[3])
            / (f0 * k * a[3] + one);
        // since we know the feedback, we can solve the remaining outputs:
        self.vout[0] = g0 * (g * a[1] * (input * a[0] - k * a[3] * self.vout[3]) + self.s[0]);
        self.vout[1] = g1 * (g * a[2] * self.vout[0] + self.s[1]);
        self.vout[2] = g2 * (g * a[3] * self.vout[1] + self.s[2]);

        self.vout[self.params.slope.value() as usize]
    }
    // linear version without distortion
    pub fn run_filter_linear(&mut self, input: f32x4) -> f32x4 {
        // denominators of solutions of individual stages. Simplifies the math a bit
        let g = f32x4::splat(self.params.g.get());
        let k = f32x4::splat(self.params.k_ladder.get());
        let one = f32x4::splat(1.);
        let g0 = one / (one + g);
        let g1 = g * g0 * g0;
        let g2 = g * g1 * g0;
        let g3 = g * g2 * g0;
        // outputs a 24db filter
        self.vout[3] =
            (g3 * g * input + g0 * self.s[3] + g1 * self.s[2] + g2 * self.s[1] + g3 * self.s[0])
                / (g3 * g * k + one);
        // since we know the feedback, we can solve the remaining outputs:
        self.vout[0] = g0 * (g * (input - k * self.vout[3]) + self.s[0]);
        self.vout[1] = g0 * (g * self.vout[0] + self.s[1]);
        self.vout[2] = g0 * (g * self.vout[1] + self.s[2]);
        self.vout[self.params.slope.value() as usize]
    }
    pub fn run_filter_newton(&mut self, input: f32x4) -> f32x4 {
        // ---------- setup ----------
        // load in g and k from parameters
        let g = f32x4::splat(self.params.g.get());
        let k = f32x4::splat(self.params.k_ladder.get());
        // a[n] is the fixed-pivot approximation for whatever is being processed nonlinearly
        let mut v_est: [f32x4; 4];
        let mut temp: [f32x4; 4] = [f32x4::splat(0.); 4];

        // use state as estimate
        v_est = [self.s[0], self.s[1], self.s[2], self.s[3]];

        let mut tanh_input = tanh_levien(input - k * v_est[3]);
        let mut tanh_y1_est = tanh_levien(v_est[0]);
        let mut tanh_y2_est = tanh_levien(v_est[1]);
        let mut tanh_y3_est = tanh_levien(v_est[2]);
        let mut tanh_y4_est = tanh_levien(v_est[3]);
        let mut residue = [
            g * (tanh_input - tanh_y1_est) + self.s[0] - v_est[0],
            g * (tanh_y1_est - tanh_y2_est) + self.s[1] - v_est[1],
            g * (tanh_y2_est - tanh_y3_est) + self.s[2] - v_est[2],
            g * (tanh_y3_est - tanh_y4_est) + self.s[3] - v_est[3],
        ];
        // println!("residue: {:?}", residue);
        // println!("vest: {:?}", v_est);
        // let max_error = 0.00001;
        let max_error = f32x4::splat(0.00001);
        let mut n_iterations = 0;

        // f32x4.lt(max_error) returns a mask.
        while residue[0].abs().simd_gt(max_error).any()
            || residue[1].abs().simd_gt(max_error).any()
            || residue[2].abs().simd_gt(max_error).any()
            || residue[3].abs().simd_gt(max_error).any()
        // && n_iterations < 9
        {
            n_iterations += 1;
            if n_iterations > 9 {
                println!("TOO MANY ITS");
            }
            let one = f32x4::splat(1.);
            // jacobian matrix
            let j10 = g * (one - tanh_y1_est * tanh_y1_est);
            let j00 = -j10 - one;
            let j03 = -g * k * (one - tanh_input * tanh_input);
            let j21 = g * (one - tanh_y2_est * tanh_y2_est);
            let j11 = -j21 - one;
            let j32 = g * (one - tanh_y3_est * tanh_y3_est);
            let j22 = -j32 - one;
            let j33 = -g * (one - tanh_y4_est * tanh_y4_est) - one;

            // this one is disgustingly huge, but couldn't find a way to avoid that. Look into inverting matrix
            // maybe try replacing j_m_n with the expressions and simplify in maple? <- didn't help
            temp[0] = (((j22 * residue[3] - j32 * residue[2]) * j11
                + j21 * j32 * (-j10 * v_est[0] + residue[1]))
                * j03
                + j11 * j22 * j33 * (j00 * v_est[0] - residue[0]))
                / (j00 * j11 * j22 * j33 - j03 * j10 * j21 * j32);

            temp[1] = (j10 * v_est[0] - j10 * temp[0] + j11 * v_est[1] - residue[1]) / (j11);
            temp[2] = (j21 * v_est[1] - j21 * temp[1] + j22 * v_est[2] - residue[2]) / (j22);
            temp[3] = (j32 * v_est[2] - j32 * temp[2] + j33 * v_est[3] - residue[3]) / (j33);

            v_est = temp;
            tanh_input = tanh_levien(input - k * v_est[3]);
            tanh_y1_est = tanh_levien(v_est[0]);
            tanh_y2_est = tanh_levien(v_est[1]);
            tanh_y3_est = tanh_levien(v_est[2]);
            tanh_y4_est = tanh_levien(v_est[3]);

            residue = [
                g * (tanh_input - tanh_y1_est) + self.s[0] - v_est[0],
                g * (tanh_y1_est - tanh_y2_est) + self.s[1] - v_est[1],
                g * (tanh_y2_est - tanh_y3_est) + self.s[2] - v_est[2],
                g * (tanh_y3_est - tanh_y4_est) + self.s[3] - v_est[3],
            ];
            // n_iterations += 1;
        }
        self.vout = v_est;
        self.vout[self.params.slope.value() as usize]
    }
    // performs a complete filter process (newton-raphson method)
    pub fn tick_newton(&mut self, input: f32x4) -> f32x4 {
        // perform filter process
        let out = self.run_filter_newton(input * f32x4::splat(self.params.drive.value + 1.));
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out * f32x4::splat((1. + self.params.k_ladder.get()) / (self.params.drive.value * 0.5 + 1.))
    }
    // performs a complete filter process (newton-raphson method)
    pub fn tick_pivotal(&mut self, input: f32x4) -> f32x4 {
        // perform filter process
        let out = self.run_filter_pivotal(input * f32x4::splat(self.params.drive.value + 1.));
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out
    }
    // performs a complete filter process (newton-raphson method)
    pub fn tick_linear(&mut self, input: f32x4) -> f32x4 {
        // perform filter process
        // let out = self.run_filter_linear(input * f32x4::splat(self.params.drive.value + 1.));
        let out = self.run_filter_linear(input);
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out
    }
}

pub struct NewSVF {
    pub params: Arc<FilterParams>,
    vout: [f32; 2],
    s: [f32; 2],


    // used to find the nonlinear contributions
    dq: [[f32; 2]; 2],
    eq: [f32; 2],
    fq: [[f32; 4]; 8],
    // dq, eq are actually much larger, pexps are used to reduce them
    pexps: [[f32; 2]; 8],

    // used to update the capacitor states
    a: [[f32; 2]; 2],
    b: [f32; 2],
    c: [[f32; 4]; 2],

    // used to find the output values
    dy: [[f32; 2]; 2],
    ey: [f32; 2],
    fy: [[f32; 4]; 2],

    solver: DKSolver,

}

impl NewSVF {


    pub fn new(params: Arc<FilterParams>) -> Self {
        let pexps = [[ 0.        ,  1.        ],
        [ 0.34013605, -0.34013605],
        [ 1.        ,  0.        ],
        [ 0.        , -0.22675737],
        [-1.        ,  0.        ],
        [ 0.        , -1.        ],
        [ 1.        ,  0.        ],
        [ 0.        ,  1.        ]];
        let fq = [[ 0.  ,  0.  ,  1.  ,  0.  ],
        [ 8.82,  0.  ,  0.  ,  0.  ],
        [-1.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  8.82,  0.  ,  0.  ],
        [ 1.  ,  0.  ,  0.  ,  0.  ],
        [-3.  , -2.  ,  1.  ,  1.  ],
        [-1.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  1.  ]];
        Self {
            params,
            vout: [0.; 2],
            s: [0.; 2],

            dq: [[-9589.77913964,   242.39343474],
                [9428.18351647,  6527.84911239]],
            eq: [-0.01211967, -0.32639246],
            fq,
            pexps,

            a: [[ 0.91795583, -0.04847869],
            [-0.04847869,  0.96643451]],
            b: [7.27180304e-06, 1.67827426e-06],
            c:  [[0.0002, 0.    , 0.    , 0.    ],
                [0.    , 0.0002, 0.    , 0.    ]],

            dy: [[  242.39343474, -9832.17257438],
                [-9589.77913964,   242.39343474]],
            ey: [-0.00839137, -0.01211967],
            fy: [[ 0., -1.,  0.,  0.],
                [-1.,  0.,  0.,  0.]],

            solver: DKSolver::new([0.; N_P], [0.; N_N], &pexps, fq)
        }
    }


    pub fn tick_dk(&mut self, input: f32) -> f32x4 {
        // let p = dot(dq, s) + dot(eq, input);
        let mut p = [0.; 2];
        // find the
        p[0] = self.dq[0][0] * self.s[0] + self.dq[0][1] * self.s[1] +  self.eq[0] * input;
        p[1] = self.dq[1][0] * self.s[0] + self.dq[1][1] * self.s[1] +  self.eq[1] * input;
        
        // 
        self.nonlinear_contribs(p);

        // self.vout = dot(dy, s) + dot(ey, input) + dot(fy, self.solver.z)
        // TODO: add in fy * z
        self.vout[0] = self.dy[0][0] * self.s[0] + self.dy[0][1] * self.s[1] +  self.ey[0] * input;
        self.vout[1] = self.dy[1][0] * self.s[0] + self.dy[1][1] * self.s[1] +  self.ey[1] * input;
        for i in 0..2 {
            for j in 0..4 {
                self.vout[i] +=  self.solver.z[j] * self.fy[i][j];
            }
        }
        let s1_update = self.a[1][0] * self.s[0] + self.a[1][1] * self.s[1];
        self.s[0] = self.a[0][0] * self.s[0] + self.a[0][1] * self.s[1] +  self.b[0] * input;
        self.s[1] = s1_update + self.b[1] * input;
        // THE ERROR IS IN THIS FOR LOOP!!!
        // correct formula: self.s[1] = self.a[1][0] * self.s[0] + self.a[1][1] * self.s[1] +  self.b[1] * input + self.solver.z[1] * self.c[1][1]
        for i in 0..2 {
            for j in 0..4 {
                self.s[i] +=  self.solver.z[j] * self.c[i][j];
            }
        }
        // let out = self.get_output(input, self.params.zeta.get());
        let out = self.vout[0];
        f32x4::from_array([out, out, 0., 0.])
        // self.s = dot(a, s) + dot(b, input) + dot(c, self.solver.z);
    }

    fn nonlinear_contribs(&mut self, p: [f32; 2]) {
        self.solver.set_p(p, &self.pexps);

        // self.solver.tmp_np[0] =  self.solver.tmp_np[0] - self.solver.last_p[0];
        // self.solver.tmp_np[1] = self.solver.tmp_np[1] - self.solver.last_p[1];
        self.solver.tmp_np[0] = p[0] - self.solver.last_p[0];
        // FIXME: slight discrepancy in last_p[1], which seems to "spread out" to make convergence fail
        self.solver.tmp_np[1] = p[1] - self.solver.last_p[1];

        for i in 0..4 {
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
        // verified above this line
        let mut resmaxabs = 1.;
        for _plsconverge in 0..50 {
            self.solver.evaluate_nonlinearities(self.solver.z, self.fq, 0.6293835652464929/*self.params.g.get() * 0.0001 * 2. * 44100.*/);

            let maybe_resmaxabs = self.solver.residue.iter().max_by(|x, y| x.abs().partial_cmp(&y.abs()).expect(&format!("shit: {:?}", self.solver.residue)));
            if let Some(resm) = maybe_resmaxabs {
                resmaxabs = *resm;
            }   
            else {
                panic!("how does it fail with this residue?: {:?}",self.solver.residue)
            }

            self.solver.set_lin_solver(self.solver.j);
            if resmaxabs.abs() < 1e-4 {
                break;
            }

            // update z with the linsolver according to the residue
            self.solver.tmp_nn = self.solver.solve_linear_equation(self.solver.residue);

            for i in 0..self.solver.z.len() {
                self.solver.z[i] = self.solver.z[i] - self.solver.tmp_nn[i];
            }
            // self.solver.z = self.solver.z - self.solver.tmp_nn;

        }
        if resmaxabs.abs() < 1e-4 {
            self.solver.set_jp(&self.pexps);
            self.solver.set_extrapolation_origin(p, self.solver.z, self.solver.jp);
        }
        else {
            panic!("failed to converge. residue: {:?}", self.solver.residue);
        }

    }
    // TODO: I've somehow managed to swap the outputs, so just swap them here maybe? Or take the time to flip a bunch of matrices
    #[inline(always)]
    fn get_output(&self, input: f32, k: f32) -> f32 {
        match self.params.mode.value() {
            SvfMode::LP => self.vout[1],                            // lowpass
            SvfMode::HP => input - k * self.vout[0] - self.vout[1], // highpass
            SvfMode::BP1 => self.vout[0],                           // bandpass
            SvfMode::Notch => input - k * self.vout[0],             // notch
            //3 => input - 2. * k * self.vout[1], // allpass
            SvfMode::BP2 => k * self.vout[0], // bandpass (normalized peak gain)
                                              // _ => input - f32x4::splat(2.) * self.vout[1] - k * self.vout[0], // peak / resonator thingy
        }
    }

}

/// solves the nonlinear contributions in the SVF using Newton's method
// TODO: could be made to work with multiple circuits with constructing with n_n, n_p and moving eval_nonlinear to a trait
const N_N: usize = 4;
const N_P: usize = 2;
struct DKSolver {
// struct DKSolver<const n_n: usize, const n_p: usize> {
    // current solution of nonlinear contributions
    z: [f32; N_N],
    last_z: [f32; N_N],
    last_p: [f32; N_P],

    // last value of jacobian * p?
    last_jp: [[f32; N_P]; N_N],

    // temporary storage. Evaluate if necessary
    tmp_nn: [f32; N_N],
    tmp_np: [f32; N_P],

    // TODO: how to store the linear solver and the Nleq

    // used by the linearization
    factors: [[f32; N_N]; N_N],
    // indices for pivot columns for linearization
    ipiv: [usize; N_N],

    // used by the nonlinear equations

    // full jacobian for the circuit
    j: [[f32; N_N]; N_N],
    // full jacobian product for the circuit
    jp: [[f32; N_P]; N_N],
    // TODO: rename these 2, to p and Jq i think but verify
    /// was called scratch0 before
    // TODO: FIXME: these 2 are the wrong size! Needs to be 8 and 4, 8 why? <- because of p but how to define/explain
    // p: [f32; N_P],
    // jq: [[f32; N_N]; N_P],
    p_full: [f32; N_N * N_P],
    // jq: [[f32; N_N]; N_N * N_P],
    jq: [[f32; N_N * N_P]; N_N],
    residue: [f32; N_N],


}

impl DKSolver {

    fn new(initial_p: [f32; N_P], initial_z: [f32; N_N], pexps: &[[f32; 2];N_N * N_P], fq: [[f32; N_N]; N_N * N_P]) -> Self {
        let mut a = Self {
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
            p_full: [0.; N_N * N_P],
            jq: [[0.; N_N * N_P]; N_N],
            residue: [0.; N_N],
        };
        a.set_p(initial_p, pexps);
        a.evaluate_nonlinearities(initial_z, fq, 0.6293835652464929);
        a.set_lin_solver(a.j);
        a.set_jp(pexps);
        a.set_extrapolation_origin(initial_p, initial_z, a.jp);
        a
    }

    fn set_p(&mut self, p: [f32; 2], pexps: &[[f32; 2];N_N * N_P]) {
        self.p_full = [0.; N_N * N_P];
        for i in 0..8 {
            for j in 0..2 {
                self.p_full[i] += p[j] * pexps[i][j];
            }
        }
    }

    fn set_jp(&mut self, pexps: &[[f32; 2];N_N * N_P]) {
        // goal shape: (4,2)
        // np.dot(self.jq, pexps)
        // not verified but I think it's right
        for i in 0..4 {
            for j in 0..2 {
                self.jp[i][j] = 0.;
                for k in 0..8 {
                    self.jp[i][j] += self.jq[i][k] * pexps[k][j];
                }
            }
        }
    }

    // prepare the solver for next sample by storing the current solution, so it can be used as an initial guess
    // NOTE: this generally works very well but can lead to slow convergence on sudden discontinuities, e.g. the jump in a saw wave
    // In that case maybe a guess from the capacitor states would be better
    fn set_extrapolation_origin(&mut self, p: [f32; N_P], z: [f32; 4], jp: [[f32; 2]; 4]) {

        self.last_jp = jp;
        self.last_p = p;
        self.last_z = z;
    }
    // this entire function could be removed by just using a linearization directly but it would make updating the model(s) require a lot more manual work
    fn set_lin_solver(&mut self, new_jacobian: [[f32; N_N]; N_N]) {
        const M: usize = N_N;
        const N: usize = N_N;

        self.factors = new_jacobian;
        // sort of a lower-upper factorization
        for k in 0..N_N {
            let mut kp = k;
            let mut amax = 0.0;
            for i in k..M {
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
                    for i in 0..N {
                        let tmp = self.factors[k][i];
                        self.factors[k][i] = self.factors[kp][i];
                        self.factors[kp][i] = tmp;
                    }
                }
                // scale first column
                // let fkk_inv =  1. / self.factors[k][k];
                self.factors[k][k] = 1. / self.factors[k][k];
                for i in k+1..M {
                    self.factors[i][k] *= self.factors[k][k];
                }

            }
            else {
                panic!("shouldn't happen");
            }
            // update rest of factors
            for j in k+1..N {
                for i in k+1..M {
                    self.factors[i][j] -= self.factors[i][k] * self.factors[k][j];
                }
            }
        }
    }
    /// based on dgetrs, solve A * X = B with A being lower-upper factorized
    fn solve_linear_equation(&self, x: [f32; N_N]) -> [f32; N_N] {
        let mut x_temp = x;
        for i in 0..N_N {
            // x[i], x[self.ipiv[i]] =
            x_temp.swap(i, self.ipiv[i]);
        }
        for j in 0..N_N {
            let xj = x_temp[j];
            for i in j+1..N_N {
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

    fn evaluate_nonlinearities(&mut self, z: [f32; N_N], fq: [[f32; N_N]; N_N * N_P], g: f32) {
        // TODO: better way of finding dot-product between fq and z
        let mut dot_p = [0.; 8];
        let mut q = self.p_full;
        for i in 0..8 {
            for j in 0..4 {
                dot_p[i] += z[j] * fq[i][j];
            }
            q[i] += dot_p[i];
        }
        // let q = self.p_full + dot_p;
        // println!("g: {g}");
        // println!("q: {:?}", q);
        // println!("p_full: {:?}", self.p_full);
        // println!("z: {:?}", z);
        let (res1, jq1) = self.eval_ota(&q[0..2], g);
        let (res2, jq2) = self.eval_ota(&q[2..4], g);

        let (res3, jq3) = self.eval_diode(&q[4..6]);
        let (res4, jq4) = self.eval_diode(&q[6..8]);

        // TODO: consider simplifying jq
        self.jq[0][0] = jq1[0];
        self.jq[0][1] = jq1[1];

        self.jq[1][2] = jq2[0];
        self.jq[1][3] = jq2[1];

        self.jq[2][4] = jq3[0];
        self.jq[2][5] = jq3[1];
        self.jq[3][6] = jq4[0];
        self.jq[3][7] = jq4[1];

        // update j to the matrix product fq * jq
        for i in 0..self.jq.len() {
            for j in 0..N_N {
                self.j[i][j] = 0.;
                for k in 0..8 {
                    self.j[i][j] += self.jq[i][k] * fq[k][j];
                }
            }
        }
        self.residue = [res1, res2, res3, res4];

    }

    // TODO: remove g from this when switching to the new analytic matrices
    fn eval_ota(&self, q: &[f32], g: f32)  -> (f32, [f32; 2]) {
        let v_in = q[0];
        let i_out = q[1];
        // TODO: switch to tanh approximation
        let tanh_vin = v_in.tanh();
        let residue = g * tanh_vin + i_out;

        let jacobian = [g * (1. - tanh_vin * tanh_vin), 1.0];

        (residue, jacobian)
    }
    // simple shockley diode equation
    fn eval_diode(&self, q: &[f32])  -> (f32, [f32; 2]) {
        // thermal voltage
        const V_T_INV: f32 = 1.0/25e-3;
        // the diode's saturation current. Could make this a function parameter to have slightly mismatched diodes or something
        const I_S: f32 = 1e-15;

        let v_in = q[0];
        let i_out = q[1];
        let ex = (v_in * V_T_INV).exp();

        let residue = I_S * (ex - 1.) - i_out;

        let jacobian = [I_S * V_T_INV * ex, -1.0];

        (residue, jacobian)
    }

}



// this is a 2-pole filter with resonance, which is why there's 2 states and vouts
pub struct SVF {
    pub params: Arc<FilterParams>,
    vout: [f32x4; 2],
    s: [f32x4; 2],
}
#[allow(dead_code)]
impl SVF {
    pub fn new(params: Arc<FilterParams>) -> Self {
        Self {
            params,
            vout: [f32x4::splat(0.); 2],
            s: [f32x4::splat(0.); 2],
        }
    }
    // the state needs to be updated after each process. Found by trapezoidal integration
    #[inline]
    fn update_state(&mut self) {
        let two = f32x4::splat(2.);
        self.s[0] = two * self.vout[0] - self.s[0];
        self.s[1] = two * self.vout[1] - self.s[1];
    }
    fn get_estimate(&mut self, n: usize, estimate: EstimateSource, input: f32x4) -> f32x4 {
        // if we ask for an estimate based on the linear filter, we have to run it
        if estimate == EstimateSource::LinearStateEstimate
            || estimate == EstimateSource::LinearVoutEstimate
        {
            self.run_svf_linear(input);
        }
        match estimate {
            EstimateSource::State => self.s[n],
            EstimateSource::PreviousVout => self.vout[n],
            EstimateSource::LinearStateEstimate => f32x4::splat(2.) * self.vout[n] - self.s[n],
            EstimateSource::LinearVoutEstimate => self.vout[n],
        }
    }
    // performs a complete filter process (fixed-pivot method)
    fn tick_pivotal(&mut self, input: f32x4) -> f32x4 {
        // perform filter process
        let out = self.run_svf_pivotal(input * f32x4::splat(self.params.drive.value + 1.));
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out
    }
    // performs a complete filter process (Newton's method)
    pub fn tick_newton(&mut self, input: f32x4) -> f32x4 {
        // perform filter process
        let mut out = self.run_svf_newton(input * f32x4::splat(self.params.drive.value + 1.));
        // staturating the output and adding some gain compensation for drive
        // should be similar to the EDP Wasp filter
        // TODO: check if tanh distortion feels too strong. If so, implement a simd_asinh or smth
        out = tanh_levien(out * f32x4::splat(0.5))
            * f32x4::splat(2. / (self.params.drive.value * 0.5 + 1.));
        // update ic1eq and ic2eq for next sample
        self.update_state();

        out
    }
    pub fn run_svf_linear(&mut self, input: f32x4) -> f32x4 {
        let g = f32x4::splat(self.params.g.get());
        // let k = self.params.res.get();
        let k = f32x4::splat(self.params.zeta.get());
        let one = f32x4::splat(1.);
        // declaring some constants that simplifies the math a bit
        let g1 = one / (one + g * (g + k));
        let g2 = g * g1;
        // let g3 = g * g2;
        // find the correct output voltages
        self.vout[0] = g1 * self.s[0] + g2 * (input - self.s[1]);
        // self.vout[1] = (input - self.s[1]) * g3 + self.s[0] * g2 + self.s[1]; <- meant for parallel processing
        self.vout[1] = self.s[1] + g * self.vout[0];
        // get output according to the current filter_mode
        self.get_output(input, k)
    }
    pub fn run_svf_pivotal(&mut self, input: f32x4) -> f32x4 {
        // ---------- setup ----------
        // load in g and k from parameters
        let g = f32x4::splat(self.params.g.get());
        // let k = self.params.res.get();
        let k = f32x4::splat(self.params.zeta.get());
        let one = f32x4::splat(1.);

        // a[n] is the fixed-pivot approximation for whatever is being processed nonlinearly
        let mut a = [f32x4::splat(1.); 3];
        let est_type = EstimateSource::State;
        // first getting fixed-pivot approximation for the feedback line, since it's necessary for computing a[0]:
        let est_source_a2 = self.get_estimate(0, est_type, input);
        // employing fixed-pivot method
        // a[2] first, since it involves the antisaturator
        let mask1 = est_source_a2.simd_ne(f32x4::splat(0.));
        // a2 is clipped with the inverse of the diode anti-saturator
        // a[2] = (v_t * simd_asinh(est_source_a2 / i_s)) / est_source_a2;
        a[2] = simd_asinh(est_source_a2) / est_source_a2;
        // using the mask to remove any elements that have had division by 0
        a[2] = mask1.select(a[2], f32x4::splat(1.));
        let est_source_rest = [
            (input
                - (est_source_a2 * a[2] + (k - one) * est_source_a2)
                - self.get_estimate(1, est_type, input)),
            self.get_estimate(0, est_type, input),
        ];
        for n in 0..est_source_rest.len() {
            let mask = est_source_rest[n].simd_ne(f32x4::splat(0.));
            a[n] = tanh_levien(est_source_rest[n]) / est_source_rest[n];
            // since the line above can become NaN or other stuff when a value in base[n] is 0,
            // replace values where a[n] is 0.
            a[n] = mask.select(a[n], f32x4::splat(1.));
        }
        // ---------- calculations ----------
        // factored out of the equation
        let g1 = one / (g * a[0]);
        let g2 = one / (a[0] * a[2] * g * g1 * k - a[0] * a[2] * g * g1 + a[2] * g1 + one);
        let g3 = one / (one + g * g * a[0] * a[1] * g1 * g2 * a[2]);
        // solving equations for output voltages at v1 and v2
        let u = (g * a[0] * input - g * a[0] * self.s[1] + self.s[0]) * g1 * g2 * g3;
        self.vout[0] = simd_asinh(u);
        self.vout[1] = g * a[1] * self.vout[0] + self.s[1];
        // here, the output is chosen to give the specified type of filter
        self.get_output(input, k)
    }
    // trying to avoid having to invert the matrix
    pub fn run_svf_newton(&mut self, input: f32x4) -> f32x4 {
        // ---------- setup ----------
        // load in g and k from parameters
        let g = f32x4::splat(self.params.g.get());
        // potentially useful knowledge: filter starts self-oscillating at about k = -0.001
        let k = f32x4::splat(self.params.zeta.get());
        let one = f32x4::splat(1.);
        // let k = self.params.res.get();
        // a[n] is the fixed-pivot approximation for whatever is being processed nonlinearly
        let mut v_est: [f32x4; 2];
        let est_type = EstimateSource::State;
        // let est_type = EstimateSource::State;

        // getting initial estimate. Could potentially be done with the fixed_pivot filter
        v_est = [
            self.get_estimate(0, est_type, input),
            self.get_estimate(1, est_type, input),
        ];
        // let mut sinh_v_est0 = v_est[0].sinh();
        // let mut cosh_v_est0 = v_est[0].cosh();
        let mut tanh_v_est0 = tanh_levien(v_est[0]);
        let mut cosh_v_est0 = simd_cosh(v_est[0]);
        let mut sinh_v_est0 = tanh_v_est0 * cosh_v_est0; // from a trig identity
        let mut fb_line = tanh_levien(input - ((k - one) * v_est[0] + sinh_v_est0) - v_est[1]);
        // using fixed_pivot as estimate
        // self.run_svf_pivotal(input);
        // v_est = [self.vout[0], self.vout[1]];
        let mut residue = [
            g * fb_line + self.s[0] - v_est[0],
            g * tanh_v_est0 + self.s[1] - v_est[1],
        ];

        let max_error = f32x4::splat(0.00001);
        let mut n_iterations = 0;
        while (residue[0].abs().simd_gt(max_error).any()
            || residue[0].abs().simd_gt(max_error).any())
            && n_iterations < 9
        {
            // terminate if error doesn't improve after 10 iterations
            // if n_iterations > 9 {
            //     // panic!("infinite loop mayhaps?");
            //     break;
            // }
            let new_bigboy = one - fb_line * fb_line;
            // jacobian matrix
            let j00 = -g * new_bigboy * (k - one + cosh_v_est0) - one;
            let j01 = -g * new_bigboy;
            let j10 = g * (one - tanh_v_est0 * tanh_v_est0);

            v_est[0] = (j01 * j10 * v_est[0] + j00 * v_est[0] - j01 * residue[1] - residue[0])
                / (j01 * j10 + j00);
            v_est[1] = (j01 * j10 * v_est[1] + j00 * residue[1] + j00 * v_est[1]
                - j10 * residue[0])
                / (j01 * j10 + j00);
            cosh_v_est0 = simd_cosh(v_est[0]);
            tanh_v_est0 = tanh_levien(v_est[0]);
            sinh_v_est0 = tanh_v_est0 * cosh_v_est0;
            fb_line =
                tanh_levien(input - ((k - f32x4::splat(1.)) * v_est[0] + sinh_v_est0) - v_est[1]);
            // recompute filter
            // residue = self.run_helper_svf(g, tanh_v_est0, fb_line, v_est);
            residue = [
                g * fb_line + self.s[0] - v_est[0],
                g * tanh_v_est0 + self.s[1] - v_est[1],
            ];
            n_iterations += 1;
        }
        // when newton's method is done, we have some good estimates for vout
        self.vout = v_est;
        // here, the output is chosen to give the specified type of filter
        self.get_output(input, k)
    }

    /// here the output is chosen to give the specified type of filter
    #[inline(always)]
    fn get_output(&self, input: f32x4, k: f32x4) -> f32x4 {
        match self.params.mode.value() {
            SvfMode::LP => self.vout[1],                            // lowpass
            SvfMode::HP => input - k * self.vout[0] - self.vout[1], // highpass
            SvfMode::BP1 => self.vout[0],                           // bandpass
            SvfMode::Notch => input - k * self.vout[0],             // notch
            //3 => input - 2. * k * self.vout[1], // allpass
            SvfMode::BP2 => k * self.vout[0], // bandpass (normalized peak gain)
                                              // _ => input - f32x4::splat(2.) * self.vout[1] - k * self.vout[0], // peak / resonator thingy
        }
    }
}
// impl Default for SVF {
//     fn default() -> Self {
//         Self {
//             params: Arc::new(FilterParams::new()),
//             vout: [f32x4::splat(0.); 2],
//             s: [f32x4::splat(0.); 2],
//         }
//     }
// }
// impl Default for LadderFilter {
//     fn default() -> Self {
//         Self {
//             params: Arc::new(FilterParams::default()),
//             vout: [f32x4::splat(0.); 4],
//             s: [f32x4::splat(0.); 4],
//         }
//     }
// }


#[test]
fn test() {
    let should_update_filter = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let params = Arc::new(FilterParams::new(should_update_filter.clone()));

    let mut filt = NewSVF::new(params);
    for i in 0..10 {
        println!("sample {i}");
        filt.tick_dk(-1.0);
    }
}   