#![no_std]

use cv::sample_consensus::{Consensus, Estimator};
use cv::{KeypointWorldMatch, WorldPose};

pub fn pnp<E, C, I>(estimator: E, mut consensus: C, data: I) -> Option<WorldPose>
where
    E: Estimator<KeypointWorldMatch, Model = WorldPose>,
    C: Consensus<E, KeypointWorldMatch>,
    I: Iterator<Item = KeypointWorldMatch> + Clone,
{
    // TODO: Actually refine model with Levenburg-Marquardt.
    consensus.model(&estimator, data)
}
