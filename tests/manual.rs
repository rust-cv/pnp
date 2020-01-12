use arraymap::ArrayMap;
use arrsac::{Arrsac, Config};
use cv::nalgebra::{Isometry3, Point2, Point3, Translation, UnitQuaternion, Vector3};
use cv::{KeyPointWorldMatch, WorldPoint};
use p3p::nordberg::NordbergEstimator;
use pnp::pnp;
use rand::{rngs::SmallRng, Rng, SeedableRng};

const LM_DIFF_THRESH: f64 = 1e-6;
const NOISE_LEVEL: f32 = 1e-3;
const NOISY_ITERATIONS: usize = 1024;

#[test]
fn noiseless() {
    manual_test_mutator(|p| p);
}

#[test]
fn noisy() {
    let mut rng = SmallRng::from_seed([0; 16]);
    let diff_average = (0..NOISY_ITERATIONS)
        .map(|_| {
            manual_test_mutator(|p| {
                p.coords
                    .map(|n| n + NOISE_LEVEL * (rng.gen::<f32>() - 0.5))
                    .into()
            })
        })
        .sum::<f64>()
        / NOISY_ITERATIONS as f64;
    assert!(
        diff_average > LM_DIFF_THRESH,
        "Levenberg-Marquardt was not much better than arrsac with a difference of {}",
        diff_average,
    );
}

fn manual_test_mutator(mut mutator: impl FnMut(Point2<f32>) -> Point2<f32>) -> f64 {
    // Define some points in camera coordinates (with z > 0).
    let camera_depth_points = [
        [-0.228_125, -0.061_458_334, 1.0],
        [0.418_75, -0.581_25, 2.0],
        [1.128_125, 0.878_125, 3.0],
        [-0.528_125, 0.178_125, 2.5],
        [0.514_125, -0.748_125, 1.2],
        [0.114_125, -0.247_125, 3.7],
        [-0.814_125, 0.554_125, 5.2],
        [1.514_125, -0.154_125, 4.2],
        [-0.414_125, 0.054_125, 5.0],
        [0.414_125, 0.754_125, 3.0],
        [0.314_125, -0.950_125, 3.4],
    ]
    .map(|&p| Point3::from(p));

    // Define the camera pose.
    let rot = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    let trans = Translation::from(Vector3::new(0.1, 0.2, 0.3));
    let pose = Isometry3::from_parts(trans, rot);

    // Compute world coordinates.
    let world_points = camera_depth_points.map(|p| pose.inverse() * p);

    // Compute normalized image coordinates.
    let normalized_image_coordinates = camera_depth_points.map(|p| (p / p.z).xy());
    let mut mutated_normalized_image_coordinates = normalized_image_coordinates.clone();

    // Mutate image coordinates.
    for coord in &mut mutated_normalized_image_coordinates {
        *coord = mutator(*coord);
    }

    // Create the data samples from mutated coordinates and image coordinates.
    let samples: Vec<KeyPointWorldMatch> = world_points
        .iter()
        .zip(&mutated_normalized_image_coordinates)
        .map(|(&world, &image)| KeyPointWorldMatch(image.into(), world.into()))
        .collect();

    let get_pnp_sos = |n: usize| {
        // Run pnp on the data.
        let (pose, inliers) = pnp(
            n,
            0.1,
            &NordbergEstimator,
            &mut Arrsac::new(Config::new(0.1), SmallRng::from_seed([1; 16])),
            samples.iter().copied(),
        )
        .unwrap();

        // Ensure all points were marked as inliers.
        assert_eq!(inliers.len(), camera_depth_points.len());

        // Compare the pose to ground truth.
        normalized_image_coordinates
            .iter()
            .cloned()
            .zip(world_points.iter().map(|wp| pose.project(WorldPoint(*wp))))
            .map(|(ocoord, ecoord)| (ocoord - ecoord.0).norm_squared())
            .sum::<f32>()
    };

    // Get the squared reprojection error for both 0 and 50 iterations of Levenberg-Marquardt.
    let sos0 = get_pnp_sos(0);
    let sos50 = get_pnp_sos(50);

    sos0 as f64 - sos50 as f64
}
