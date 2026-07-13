/// Represents a grid for real-valued fields (field-resolved propagation)
#[derive(Debug, Clone)]
pub struct RealGrid {
    pub zmax: f64,
    pub reference_lambda: f64,
    pub t: Vec<f64>,
    pub ω: Vec<f64>,
    pub to: Vec<f64>,
    pub ωo: Vec<f64>,
    pub sidx: Vec<bool>,
    pub ωwin: Vec<f64>,
    pub twin: Vec<f64>,
    pub towin: Vec<f64>,
}

/// Represents a grid for complex envelope fields
#[derive(Debug, Clone)]
pub struct EnvGrid {
    pub zmax: f64,
    pub reference_lambda: f64,
    pub ω0: f64,
    pub t: Vec<f64>,
    pub ω: Vec<f64>,
    pub to: Vec<f64>,
    pub ωo: Vec<f64>,
    pub sidx: Vec<bool>,
    pub ωwin: Vec<f64>,
    pub twin: Vec<f64>,
    pub towin: Vec<f64>,
}

impl RealGrid {
    pub fn new(
        zmax: f64,
        reference_lambda: f64,
        t: Vec<f64>,
        ω: Vec<f64>,
        to: Vec<f64>,
        ωo: Vec<f64>,
        sidx: Vec<bool>,
        ωwin: Vec<f64>,
        twin: Vec<f64>,
        towin: Vec<f64>,
    ) -> Self {
        Self {
            zmax,
            reference_lambda,
            t,
            ω,
            to,
            ωo,
            sidx,
            ωwin,
            twin,
            towin,
        }
    }
}
