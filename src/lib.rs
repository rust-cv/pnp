#![no_std]

extern crate alloc;
use alloc::vec::Vec;
use cv::sample_consensus::{Consensus, Estimator};
use cv::{KeyPointWorldMatch, WorldPose};
use nalgebra::{
    dimension::{Dynamic, U2},
    Matrix, VecStorage,
};

/// Perspective-n-point algorithm
///
/// - `max_iterations` - The number of iterations Levenberg-Marquardt will be attempted before ending
///   - This may need to be higher if `convergence_speed` is lower.
/// - `convergence_speed` - The speed at which Levenberg-Marquardt will attempt to converge
///   - This number should be less than `1.0` or it probably wont improve. `0.01` is fairly good.
pub fn pnp<E, C, I>(
    max_iterations: usize,
    convergence_speed: f32,
    estimator: E,
    mut consensus: C,
    data: I,
) -> Option<WorldPose>
where
    E: Estimator<KeyPointWorldMatch, Model = WorldPose>,
    C: Consensus<E, KeyPointWorldMatch>,
    I: Iterator<Item = KeyPointWorldMatch> + Clone,
{
    consensus
        .model_inliers(&estimator, data.clone())
        .map(|(pose, inliers)| {
            let inliers: Vec<usize> = inliers.into_iter().collect();
            WorldPose::from_vec(levenberg_marquardt::optimize(
                max_iterations,
                5,
                50.0,
                0.8,
                2.0,
                0.0,
                pose.to_vec(),
                |v| {
                    let pose = WorldPose::from_vec(v);
                    pose.to_vec()
                },
                |v| {
                    let pose = WorldPose::from_vec(*v);
                    let data: Vec<f32> = inliers
                        .iter()
                        .map(|&n| data.clone().nth(n).unwrap())
                        .flat_map(|kpwm| {
                            use core::iter::once;
                            let pe = pose.projection_error(kpwm);
                            once(pe.x).chain(once(pe.y))
                        })
                        .map(|v| convergence_speed * v)
                        .collect();
                    let storage = VecStorage::new(U2, Dynamic::new(inliers.len()), data);
                    Matrix::from_data(storage)
                },
                |v| {
                    let pose = WorldPose::from_vec(*v);
                    inliers
                        .iter()
                        .map(|&n| data.clone().nth(n).unwrap())
                        .map(move |KeyPointWorldMatch(_, wp)| pose.projection_pose_jacobian(wp))
                },
            ))
        })
}
